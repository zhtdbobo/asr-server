"""Transcribe a local audio folder using remote diarization and ASR services."""

import argparse
import json
import logging
import math
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from traceback import format_exception
from uuid import uuid4

from config import Settings
from pipeline import MeetingPipeline, validate_options

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {
    ".wav",
    ".wave",
    ".mp3",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".amr",
    ".aif",
    ".aiff",
    ".caf",
    ".mka",
}


@contextmanager
def batch_logging(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    root = logging.getLogger()
    previous_level = root.level
    # In CLI use, write to disk before the console, which may be paused by selection.
    handlers = [file_handler]
    if not root.handlers:
        handlers.append(logging.StreamHandler())
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        yield
    finally:
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()
        root.setLevel(previous_level)


def comparison_segments(result: dict | list) -> list[dict]:
    """Use the same four-field array as the TextGrid reference export."""
    records = result.get("segments") if isinstance(result, dict) else result
    if not isinstance(records, list):
        raise TypeError("识别结果缺少 segments 数组")
    segments = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("片段必须是包含 speaker/start/end/text 的对象")
        try:
            speaker, text = record["speaker"], record["text"]
            start, end = record["start"], record["end"]
        except KeyError as exc:
            raise ValueError(f"片段缺少字段: {exc.args[0]}") from exc
        if not isinstance(speaker, str) or not speaker.strip() or not isinstance(text, str):
            raise ValueError("speaker 必须为非空字符串，text 必须为字符串")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (start, end)
        ):
            raise ValueError("片段起止时间必须是有限数值")
        if start < 0 or end < start:
            raise ValueError(f"片段时间区间无效: {start} -> {end}")
        if text.strip():
            segments.append({"speaker": speaker, "start": start, "end": end, "text": text.strip()})
    return sorted(
        segments, key=lambda segment: (segment["start"], segment["end"], segment["speaker"])
    )


def discover_audio(folder: Path, output_dir: Path, recursive: bool) -> list[Path]:
    folder, output_dir = folder.resolve(), output_dir.resolve()
    if not folder.is_dir():
        raise ValueError(f"音频文件夹不存在或不是目录: {folder}")
    exclude_output = output_dir != folder and output_dir.is_relative_to(folder)
    paths = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        (
            path
            for path in paths
            if path.is_file()
            and path.suffix.lower() in AUDIO_EXTENSIONS
            and not (exclude_output and path.resolve().is_relative_to(output_dir))
        ),
        key=lambda path: path.relative_to(folder).as_posix().casefold(),
    )


def write_outputs(files: dict[Path, str]):
    """Stage complete files before replacing results; remove partial temporary files."""
    staged = []
    try:
        for target, content in files.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
            staged.append((temporary, target))
            temporary.write_text(content, encoding="utf-8")
        for temporary, target in staged:
            temporary.replace(target)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def run_batch(
    folder: Path,
    output_dir: Path,
    pipeline,
    *,
    workers: int = 1,
    recursive: bool = False,
    skip_existing: bool = False,
    options: dict | None = None,
) -> dict:
    if workers < 1:
        raise ValueError("workers 必须大于 0")
    folder, output_dir = folder.resolve(), output_dir.resolve()
    files = discover_audio(folder, output_dir, recursive)
    if not files:
        raise ValueError("文件夹中没有支持的音频文件；如音频在子目录，请加 --recursive")
    options = options or {}
    started = perf_counter()
    logger.info(
        f"找到 {len(files)} 个音频，最多同时处理 {workers} 个；JSON 用于对比，TXT 用于阅读。",
    )

    def process(path):
        relative = path.relative_to(folder)
        # Retain the extension so meeting.wav and meeting.mp3 cannot overwrite each other.
        text_path = output_dir / relative.parent / f"{relative.name}.txt"
        json_path = output_dir / relative.parent / f"{relative.name}.json"
        full_json_path = output_dir / relative.parent / f"{relative.name}.full.json"
        item = {
            "file": relative.as_posix(),
            "text_file": str(text_path),
            "json_file": str(json_path),
        }
        try:
            if skip_existing and text_path.is_file() and json_path.is_file():
                try:
                    existing = json.loads(json_path.read_text(encoding="utf-8-sig"))
                    segments = comparison_segments(existing)
                except (ValueError, TypeError, OSError) as exc:
                    logger.warning("[重新处理] %s: 已有 JSON 无效（%s）", relative, exc)
                else:
                    # Migrate the old full-result JSON without repeating GPU inference.
                    updates = {}
                    if isinstance(existing, dict):
                        updates[full_json_path] = (
                            json.dumps(existing, ensure_ascii=False, indent=2) + "\n"
                        )
                    if existing != segments:
                        updates[json_path] = (
                            json.dumps(segments, ensure_ascii=False, indent=2) + "\n"
                        )
                    if updates:
                        write_outputs(updates)
                    if full_json_path.is_file():
                        item["full_json_file"] = str(full_json_path)
                    return {**item, "status": "skipped", "json_updated": bool(updates)}
            logger.info("[开始] %s", relative)
            result = pipeline.transcribe(path, **options)
            segments = comparison_segments(result)
            logger.info("[%s] 开始写入 TXT、对比 JSON 和完整 JSON", relative)
            write_outputs(
                {
                    text_path: result["formatted_text"] + "\n",
                    json_path: json.dumps(segments, ensure_ascii=False, indent=2) + "\n",
                    full_json_path: json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                }
            )
            logger.info("[%s] 三份结果文件写入完成", relative)
            return {**item, "full_json_file": str(full_json_path), "status": "completed"}
        except Exception as exc:  # noqa: BLE001 - A failed input must not abort other files.
            # Upstream errors redact secrets; don't print their raw transport causes.
            details = "".join(format_exception(type(exc), exc, exc.__traceback__, chain=False))
            logger.error("[%s] 处理失败\n%s", relative, details.rstrip())
            return {**item, "status": "failed", "error": str(exc)}

    results = []
    labels = {"completed": "成功", "failed": "失败", "skipped": "跳过"}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asr-file") as pool:
        pending = {pool.submit(process, path): path for path in files}
        while pending:
            done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            if not done:
                running = [
                    path.relative_to(folder).as_posix()
                    for future, path in pending.items()
                    if future.running()
                ]
                logger.info(
                    f"[等待] 已结束 {len(results)}/{len(files)} 个，"
                    f"本批次已耗时 {perf_counter() - started:.0f} 秒；"
                    f"正在处理: {', '.join(running) or '等待工作线程'}；"
                    "当前阶段见上方处理日志。",
                )
            for future in done:
                pending.pop(future)
                item = future.result()
                results.append(item)
                suffix = f"：{item['error']}" if item["status"] == "failed" else ""
                if item.get("json_updated"):
                    suffix += "（已有 JSON 已统一为对比格式）"
                logger.info(
                    f"[{len(results)}/{len(files)}] {labels[item['status']]} {item['file']}{suffix}",
                )

    report = {
        "input_dir": str(folder),
        "output_dir": str(output_dir),
        "total": len(files),
        "completed": sum(item["status"] == "completed" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "skipped": sum(item["status"] == "skipped" for item in results),
        "processing_seconds": round(perf_counter() - started, 3),
        "files": sorted(results, key=lambda item: item["file"].casefold()),
    }
    write_outputs(
        {output_dir / "batch-summary.json": json.dumps(report, ensure_ascii=False, indent=2) + "\n"}
    )
    return report


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="批量处理本地文件夹中的音频，生成发言人和时间戳")
    parser.add_argument("folder", type=Path, help="云桌面上的音频文件夹")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/batch"))
    parser.add_argument(
        "--log-file", type=Path, help="日志文件，默认输出目录下的 batch.log（追加）"
    )
    parser.add_argument("--recursive", action="store_true", help="同时处理子文件夹")
    parser.add_argument("--workers", type=int, default=1, help="同时处理文件数，默认 1")
    parser.add_argument(
        "--skip-existing", action="store_true", help="复用已有 TXT、JSON；旧 JSON 自动转换"
    )
    parser.add_argument(
        "--timeout-seconds", type=float, help="每次远程请求的读写等待超时秒数，默认读取 .env"
    )
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers 必须大于 0")
    log_path = (args.log_file or args.output_dir / "batch.log").resolve()
    try:
        with batch_logging(log_path):
            logger.info("批处理日志文件：%s", log_path)
            logger.info("Python：%s；版本：%s", sys.executable, sys.version.split()[0])
            return run_cli(args, parser)
    except OSError as exc:
        parser.exit(1, f"日志或输出文件无法写入: {exc}\n")


def run_cli(args, parser):
    pipeline = None
    try:
        language, counts = validate_options(
            args.language, args.num_speakers, args.min_speakers, args.max_speakers
        )
        settings = replace(Settings.from_env(), job_workers=args.workers)
        if args.timeout_seconds is not None:
            settings = replace(settings, api_timeout_seconds=args.timeout_seconds)
        if not settings.asr_api_url:
            raise ValueError("请先在 .env 中配置 ASR_API_URL")
        logger.info(
            "本次配置：%d 个文件并发，ASR 子块上限 %.1f 秒，ASR 等待上限 %.1f 秒，"
            "说话人读写等待上限 %.1f 秒",
            settings.job_workers,
            settings.asr_chunk_seconds,
            min(settings.api_timeout_seconds, settings.asr_timeout_seconds),
            settings.api_timeout_seconds,
        )
        pipeline = MeetingPipeline(settings)
        report = run_batch(
            args.folder,
            args.output_dir,
            pipeline,
            workers=args.workers,
            recursive=args.recursive,
            skip_existing=args.skip_existing,
            options={"language": language, **counts},
        )
    except (ValueError, RuntimeError, OSError) as exc:
        logger.error("批处理退出（%s）：%s", type(exc).__name__, exc)
        parser.exit(1, f"批处理失败: {exc}\n")
    finally:
        if pipeline is not None:
            pipeline.close()
    logger.info(
        f"\n处理结束：共 {report['total']} 个，成功 {report['completed']} 个，"
        f"失败 {report['failed']} 个，跳过 {report['skipped']} 个。\n"
        f"汇总：{args.output_dir.resolve() / 'batch-summary.json'}"
    )
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
