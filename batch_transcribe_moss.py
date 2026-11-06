"""Batch whole recordings through MOSS's vLLM diarized transcription API."""

import argparse
import json
import logging
import math
import mimetypes
import os
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import httpx

from batch_transcribe import (
    batch_logging,
    comparison_segments,
    create_batch_directory,
    duration_text,
    run_batch,
)
from clients import UpstreamError
from transcript import format_time

logger = logging.getLogger(__name__)
MOSS_TIMING_LABELS = {
    "inspection_seconds": "原始音频大小与时长检查",
    "request_seconds": "MOSS 上传、等待及联合推理",
    "assembly_seconds": "响应解析与结果组装",
    "write_seconds": "结果写入",
}


def inspect_audio(path: Path, max_upload_mb: float, max_audio_seconds: float) -> dict:
    """Read metadata without modifying, transcoding, or splitting the original file."""
    size = path.stat().st_size
    if size > max_upload_mb * 1024**2:
        raise UpstreamError(
            f"原文件 {size / 1024**2:.2f} MiB，超过上传上限 {max_upload_mb:g} MiB；"
            "请在服务端调整 VLLM_MAX_AUDIO_CLIP_FILESIZE_MB，"
            "并同步调整脚本的 --max-upload-mb"
        )
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise UpstreamError("未找到 ffprobe，请将 FFmpeg 的 bin 目录加入 PATH，用于检查录音时长")
    try:
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels,duration:format=duration",
                "-of",
                "json",
                str(path.resolve()),
            ],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise UpstreamError("音频时长检查超过 60 秒，请检查录音文件") from exc
    if probe.returncode:
        raise UpstreamError("无法读取音频信息，请检查文件是否包含有效音轨")
    try:
        metadata = json.loads(probe.stdout)
        stream = metadata["streams"][0]
        raw_duration = stream.get("duration")
        if raw_duration in (None, "N/A"):
            raw_duration = metadata.get("format", {}).get("duration")
        duration = float(raw_duration)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise UpstreamError("无法确定录音时长，未提交推理") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise UpstreamError("音频时长必须是大于 0 的有限秒数")
    if duration > max_audio_seconds:
        raise UpstreamError(
            f"录音时长 {duration:.3f} 秒，超过 --max-audio-seconds={max_audio_seconds:g} 秒；"
            "保留 MOSS 原生说话人编号，未自动切块"
        )
    return {
        "size_bytes": size,
        "duration_seconds": duration,
        "sample_rate": stream.get("sample_rate"),
        "channels": stream.get("channels"),
        "codec": stream.get("codec_name"),
    }


def prepare_result(body: dict) -> dict:
    """Keep the complete API response and build the existing readable TXT layout."""
    segments = comparison_segments(body)
    if not segments and body.get("text", "").strip():
        raise ValueError("返回了非空正文，但没有说话人片段，请核对 diarized_json 支持")
    duration = body.get("duration")
    if duration is not None and (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration < 0
    ):
        raise ValueError("duration 必须是非负有限秒数")
    names = {}
    paragraphs = []
    for segment in segments:
        speaker = segment["speaker"]
        if speaker not in names:
            names[speaker] = f"发言人{len(names) + 1}"
        paragraphs.append(f"{names[speaker]}\n{format_time(segment['start'])}\n{segment['text']}")
    return {
        **body,
        "formatted_text": "\n\n".join(paragraphs),
        "speaker_count": len(names),
        "speakers": [{"id": key, "name": value} for key, value in names.items()],
        "timestamp_mode": "segment",
    }


class MossPipeline:
    def __init__(
        self,
        api_url: str,
        model: str = "moss-mtd",
        *,
        api_key: str = "",
        timeout_seconds: float = 1800,
        max_completion_tokens: int = 32768,
        language: str | None = None,
        prompt: str | None = None,
        max_upload_mb: float = 500,
        max_audio_seconds: float = 5400,
        transport=None,
    ):
        parts = urlsplit(api_url)
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or not parts.path.rstrip("/").endswith("/audio/transcriptions")
        ):
            raise ValueError(
                "--api-url 需要完整接口，例如 http://服务器:8000/v1/audio/transcriptions"
            )
        if not model.strip():
            raise ValueError("--model 不能为空，须与服务的 --served-model-name 一致")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("--timeout-seconds 必须是大于 0 的有限数字")
        for name, value in (
            ("max-upload-mb", max_upload_mb),
            ("max-audio-seconds", max_audio_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"--{name} 必须是大于 0 的有限数字")
        if (
            isinstance(max_completion_tokens, bool)
            or not isinstance(max_completion_tokens, int)
            or max_completion_tokens <= 0
        ):
            raise ValueError("--max-completion-tokens 必须为正整数")
        self.api_url = api_url
        self.model = model.strip()
        self.api_key = api_key
        self.max_completion_tokens = max_completion_tokens
        self.max_upload_mb = max_upload_mb
        self.max_audio_seconds = max_audio_seconds
        self.data = {
            "model": self.model,
            "response_format": "diarized_json",
            "temperature": "0",
            "max_completion_tokens": str(max_completion_tokens),
        }
        if language:
            self.data["language"] = language
        if prompt:
            self.data["prompt"] = prompt
        self.http = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=min(15, timeout_seconds)),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            trust_env=False,
            transport=transport,
        )

    def close(self):
        self.http.close()

    def _safe_detail(self, detail) -> str:
        detail = str(detail).replace("\n", " ").replace("\r", " ")
        if self.api_key:
            detail = detail.replace(self.api_key, "[redacted]")
        return detail[:500]

    def _response_body(self, response) -> dict:
        try:
            body = response.json()
        except ValueError as exc:
            if response.is_error:
                raise UpstreamError(
                    f"MOSS 返回 HTTP {response.status_code}：{self._safe_detail(response.text)}"
                ) from exc
            raise UpstreamError("MOSS 没有返回有效 JSON") from exc
        if response.is_error:
            detail = body.get("error", body.get("detail", body)) if isinstance(body, dict) else body
            if isinstance(detail, dict):
                detail = detail.get("message", detail)
            hint = ""
            if response.status_code == 400:
                hint = "；请核对 diarized_json 支持、录音长度及 max_completion_tokens 生成上限"
            elif response.status_code == 404:
                hint = "；请核对接口路径和 --model，模型 ID 可查询 /v1/models"
            raise UpstreamError(
                f"MOSS 返回 HTTP {response.status_code}：{self._safe_detail(detail)}{hint}"
            )
        if not isinstance(body, dict):
            raise UpstreamError("MOSS 响应必须为包含 segments 的 JSON 对象")
        if body.get("error"):
            raise UpstreamError(f"MOSS 返回错误：{self._safe_detail(body['error'])}")
        if body.get("finish_reason") == "length":
            raise UpstreamError("MOSS 达到生成上限，未保存截断结果；请增大 --max-completion-tokens")
        return body

    def transcribe(self, path: Path) -> dict:
        started = perf_counter()
        timings = {}
        logger.info("[%s] 检查原始音频大小与时长", path.name)
        try:
            meta = inspect_audio(path, self.max_upload_mb, self.max_audio_seconds)
            timings["inspection_seconds"] = round(perf_counter() - started, 3)
            logger.info(
                "[%s] 检查完成，%.1f 秒，%.2f MiB，直接上传原文件",
                path.name,
                meta["duration_seconds"],
                meta["size_bytes"] / 1024**2,
            )
            result = self._transcribe_upload(path)
            result["audio_input"] = meta
            if result.get("duration") is None:
                result["duration"] = meta["duration_seconds"]
            result["timings"] = {**timings, **result["timings"]}
            result["processing_seconds"] = round(perf_counter() - started, 3)
            return result
        except UpstreamError as exc:
            if "inspection_seconds" not in timings:
                timings["inspection_seconds"] = round(perf_counter() - started, 3)
            exc.timings = {**timings, **getattr(exc, "timings", {})}
            exc.processing_seconds = round(perf_counter() - started, 3)
            raise

    def _transcribe_upload(self, path: Path) -> dict:
        started = perf_counter()
        timings = {}
        try:
            logger.info(
                "[%s] MOSS 上传整段录音 %.1f MB，model=%s，生成上限 %d tokens",
                path.name,
                path.stat().st_size / 1024**2,
                self.model,
                self.max_completion_tokens,
            )
            request_started = perf_counter()
            try:
                with path.open("rb") as audio:
                    response = self.http.post(
                        self.api_url,
                        data=self.data,
                        files={
                            "file": (
                                path.name,
                                audio,
                                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                            )
                        },
                        headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                    )
            except httpx.TransportError as exc:
                reason = "请求超时" if isinstance(exc, httpx.TimeoutException) else "连接或读取失败"
                raise UpstreamError(
                    f"MOSS {reason}；未自动重复提交，可检查服务后用 --resume-dir 续跑"
                ) from exc
            finally:
                timings["request_seconds"] = round(perf_counter() - request_started, 3)
            logger.info(
                "[%s] MOSS 响应接收完成，HTTP %d，耗时 %.1f 秒",
                path.name,
                response.status_code,
                timings["request_seconds"],
            )
            assembly_started = perf_counter()
            try:
                body = self._response_body(response)
                try:
                    result = prepare_result(body)
                except (TypeError, ValueError, AttributeError) as exc:
                    raise UpstreamError(
                        f"MOSS 结果无法转为四字段格式：{self._safe_detail(exc)}；"
                        "需要 response_format=diarized_json 返回 speaker/start/end/text"
                    ) from exc
            finally:
                timings["assembly_seconds"] = round(perf_counter() - assembly_started, 3)
            result.update(
                model=self.model,
                processing_seconds=round(perf_counter() - started, 3),
                timings=timings,
            )
            logger.info(
                "[%s] MOSS 结果组装完成，%d 个片段、%d 位说话人",
                path.name,
                len(result["segments"]),
                result["speaker_count"],
            )
            return result
        except UpstreamError as exc:
            exc.timings = dict(timings)
            exc.processing_seconds = round(perf_counter() - started, 3)
            raise


def main(argv=None):
    from dotenv import load_dotenv

    load_dotenv()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="批量调用 MOSS，输出与 Qwen 批处理相同的四字段 JSON"
    )
    parser.add_argument("folder", type=Path, help="本地音频文件夹")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("MOSS_API_URL", "http://127.0.0.1:8000/v1/audio/transcriptions"),
        help="完整的 MOSS 转写接口；也可设置 MOSS_API_URL",
    )
    parser.add_argument("--model", default=os.environ.get("MOSS_MODEL", "moss-mtd"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/moss"),
        help="输出父目录，每次新建独立的时间戳批次子目录",
    )
    parser.add_argument("--resume-dir", type=Path, help="续跑已有批次，复用有效 TXT/JSON")
    parser.add_argument("--log-file", type=Path, help="默认保存到批次目录的 batch.log")
    parser.add_argument("--recursive", action="store_true", help="包括子目录")
    parser.add_argument("--workers", type=int, default=1, help="同时处理的文件数，默认 1")
    parser.add_argument("--skip-existing", action="store_true", help="跳过当前批次的有效 TXT/JSON")
    parser.add_argument(
        "--timeout-seconds", type=float, default=1800, help="HTTP 读写等待超时，默认 1800 秒"
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=32768,
        help="每条录音的生成上限，默认 32768，须适配服务上下文长度",
    )
    parser.add_argument("--language", help="可选语言提示，例如 zh；默认由模型判断")
    parser.add_argument("--prompt", help="可选提示词或热词，传给 MOSS 接口")
    parser.add_argument(
        "--max-upload-mb",
        type=float,
        default=500,
        help="原文件上传上限 MiB，默认 500，须与服务端限制一致",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=5400,
        help="整段时长上限，默认 5400 秒；超限报错，不切块",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers 必须大于 0")
    if args.resume_dir is not None and not args.resume_dir.is_dir():
        parser.error("--resume-dir 必须指向已有的批次目录")
    pipeline = None
    try:
        pipeline = MossPipeline(
            args.api_url,
            args.model,
            api_key=os.environ.get("MOSS_API_KEY", ""),
            timeout_seconds=args.timeout_seconds,
            max_completion_tokens=args.max_completion_tokens,
            language=args.language,
            prompt=args.prompt,
            max_upload_mb=args.max_upload_mb,
            max_audio_seconds=args.max_audio_seconds,
        )
        output_dir = (
            args.resume_dir.resolve()
            if args.resume_dir is not None
            else create_batch_directory(args.output_dir)
        )
        with batch_logging((args.log_file or output_dir / "batch.log").resolve()):
            logger.info(
                "MOSS 批次目录：%s（%s）",
                output_dir,
                "续跑" if args.resume_dir is not None else "新建",
            )
            logger.info(
                "Python：%s；model=%s；并发文件数=%d", sys.executable, pipeline.model, args.workers
            )
            report = run_batch(
                args.folder,
                output_dir,
                pipeline,
                workers=args.workers,
                recursive=args.recursive,
                skip_existing=args.skip_existing or args.resume_dir is not None,
                timing_labels=MOSS_TIMING_LABELS,
            )
            logger.info(
                "处理结束：成功 %d，失败 %d，跳过 %d；批次耗时 %s；结果目录：%s",
                report["completed"],
                report["failed"],
                report["skipped"],
                duration_text(report["processing_seconds"]),
                output_dir,
            )
        return 1 if report["failed"] else 0
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"MOSS 批处理失败: {exc}\n")
    finally:
        if pipeline is not None:
            pipeline.close()


if __name__ == "__main__":
    sys.exit(main())
