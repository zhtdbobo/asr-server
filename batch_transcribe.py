"""Transcribe a local audio folder using remote diarization and ASR services."""

import argparse
import json
import logging
import math
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from traceback import format_exception
from uuid import uuid4

from clients import UpstreamError
from config import Settings
from pipeline import MeetingPipeline, validate_options

logger = logging.getLogger(__name__)
REPORT_TIMEZONE = timezone(timedelta(hours=8))
TIMING_LABELS = {
    "decode_seconds": "解码与预处理",
    "diarization_seconds": "说话人分离",
    "asr_seconds": "ASR 转写",
    "assembly_seconds": "结果组装",
    "write_seconds": "结果写入",
}

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


def report_time() -> str:
    return datetime.now(REPORT_TIMEZONE).isoformat(timespec="milliseconds")


def create_batch_directory(root: Path) -> Path:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = datetime.now(REPORT_TIMEZONE).strftime("%Y%m%d_%H%M%S")
        directory = root / f"{stamp}_{uuid4().hex[:8]}"
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        return directory


def duration_text(seconds: float | None) -> str:
    if seconds is None:
        return "未记录"
    # Round first so a value near a minute does not display as 0分60.000秒.
    milliseconds = round(seconds * 1000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    hours, minutes = divmod(minutes, 60)
    prefix = f"{hours}时{minutes:02d}分" if hours else f"{minutes}分"
    return f"{prefix}{milliseconds / 1000:06.3f}秒（{seconds:.3f} 秒）"


def timing_statistics(values) -> dict:
    samples = [value for value in values if value is not None]
    total = sum(samples)
    return {
        "count": len(samples),
        "total_seconds": round(total, 3),
        "average_seconds": round(total / len(samples), 3) if samples else None,
    }


def format_timing_report(report: dict) -> str:
    summary = report["timing_summary"]
    timing_labels = report.get("timing_labels", TIMING_LABELS)
    labels = {"completed": "成功", "failed": "失败", "skipped": "跳过"}
    lines = [
        "批次耗时报告",
        f"批次编号：{report['batch_id']}",
        f"输入目录：{report['input_dir']}",
        f"结果目录：{report['output_dir']}",
        f"开始时间：{report['started_at']}",
        f"结束时间：{report['finished_at']}",
        f"批次总耗时：{duration_text(report['processing_seconds'])}",
        (
            f"文件统计：共 {report['total']} 个，成功 {report['completed']} 个，"
            f"失败 {report['failed']} 个，跳过 {report['skipped']} 个"
        ),
        f"并发文件数：{report['workers']}",
        "",
        "耗时汇总",
    ]
    for key, title in (
        ("processed_files", "实际处理文件（含失败，不含跳过）"),
        ("completed_files", "成功文件"),
    ):
        values = summary[key]
        lines.append(
            f"{title}：统计 {values['count']} 个，累计 {duration_text(values['total_seconds'])}，"
            f"平均 {duration_text(values['average_seconds'])}"
        )
    for key, title in timing_labels.items():
        values = summary["stages"][key]
        lines.append(
            f"{title}：统计 {values['count']} 个，累计 {duration_text(values['total_seconds'])}，"
            f"平均 {duration_text(values['average_seconds'])}"
        )
    lines.extend(
        [
            "",
            "说明：批次总耗时为实际经过时间；并发文件的耗时可能重叠，累计值不等于批次总耗时。",
            (
                "说明：阶段耗时包含客户端传输、等待及推理，不是纯 GPU 耗时；"
                "未记录的数据不参与阶段平均，跳过文件不参与处理平均。"
            ),
            "",
            "逐文件明细",
        ]
    )
    for item in report["files"]:
        lines.extend(
            [
                "",
                f"文件：{item['file']}",
                f"状态：{labels[item['status']]}",
                f"开始时间：{item['started_at']}",
                f"结束时间：{item['finished_at']}",
                f"单文件总耗时：{duration_text(item['processing_seconds'])}",
                f"音频时长：{duration_text(item.get('duration_seconds'))}",
            ]
        )
        if item["status"] == "skipped":
            lines.append("  复用已有结果，本次仅记录缓存检查耗时。")
        else:
            for key, title in timing_labels.items():
                lines.append(f"  {title}：{duration_text(item.get('timings', {}).get(key))}")
        if item.get("error"):
            lines.append(f"失败原因：{item['error']}")
    return "\n".join(lines) + "\n"


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
    timing_labels: dict[str, str] | None = None,
) -> dict:
    if workers < 1:
        raise ValueError("workers 必须大于 0")
    started_at = report_time()
    started = perf_counter()
    folder, output_dir = folder.resolve(), output_dir.resolve()
    files = discover_audio(folder, output_dir, recursive)
    if not files:
        raise ValueError("文件夹中没有支持的音频文件；如音频在子目录，请加 --recursive")
    options = options or {}
    stage_labels = TIMING_LABELS if timing_labels is None else timing_labels
    logger.info(
        f"找到 {len(files)} 个音频，最多同时处理 {workers} 个；JSON 用于对比，TXT 用于阅读。",
    )

    def process(path):
        file_started = perf_counter()
        relative = path.relative_to(folder)
        # Retain the extension so meeting.wav and meeting.mp3 cannot overwrite each other.
        text_path = output_dir / relative.parent / f"{relative.name}.txt"
        json_path = output_dir / relative.parent / f"{relative.name}.json"
        full_json_path = output_dir / relative.parent / f"{relative.name}.full.json"
        item = {
            "file": relative.as_posix(),
            "text_file": str(text_path),
            "json_file": str(json_path),
            "started_at": report_time(),
        }

        def finish(outcome):
            outcome["finished_at"] = report_time()
            outcome["processing_seconds"] = round(perf_counter() - file_started, 3)
            return outcome

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
                    return finish({**item, "status": "skipped", "json_updated": bool(updates)})
            logger.info("[开始] %s", relative)
            result = pipeline.transcribe(path, **options)
            segments = comparison_segments(result)
            item.update(
                duration_seconds=result.get("duration"),
                transcription_seconds=result.get("processing_seconds"),
                timings=dict(result.get("timings", {})),
            )
            logger.info("[%s] 开始写入 TXT、对比 JSON 和完整 JSON", relative)
            writing_started = perf_counter()
            try:
                write_outputs(
                    {
                        text_path: result["formatted_text"] + "\n",
                        json_path: json.dumps(segments, ensure_ascii=False, indent=2) + "\n",
                        full_json_path: json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    }
                )
            finally:
                item["timings"]["write_seconds"] = round(perf_counter() - writing_started, 3)
            logger.info("[%s] 三份结果文件写入完成", relative)
            return finish({**item, "full_json_file": str(full_json_path), "status": "completed"})
        except Exception as exc:  # noqa: BLE001 - A failed input must not abort other files.
            if isinstance(exc, UpstreamError):
                item.update(
                    duration_seconds=getattr(exc, "duration_seconds", None),
                    transcription_seconds=getattr(exc, "processing_seconds", None),
                    timings=getattr(exc, "timings", {}),
                )
            # Upstream errors redact secrets; don't print their raw transport causes.
            details = "".join(format_exception(type(exc), exc, exc.__traceback__, chain=False))
            logger.error("[%s] 处理失败\n%s", relative, details.rstrip())
            return finish({**item, "status": "failed", "error": str(exc)})

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

    processed = [item for item in results if item["status"] != "skipped"]
    report = {
        "batch_id": output_dir.name,
        "input_dir": str(folder),
        "output_dir": str(output_dir),
        "started_at": started_at,
        "finished_at": report_time(),
        "workers": workers,
        "total": len(files),
        "completed": sum(item["status"] == "completed" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "skipped": sum(item["status"] == "skipped" for item in results),
        "processing_seconds": round(perf_counter() - started, 3),
        "timing_summary": {
            "processed_files": timing_statistics(item["processing_seconds"] for item in processed),
            "completed_files": timing_statistics(
                item["processing_seconds"] for item in processed if item["status"] == "completed"
            ),
            "stages": {
                key: timing_statistics(item.get("timings", {}).get(key) for item in processed)
                for key in stage_labels
            },
        },
        "files": sorted(results, key=lambda item: item["file"].casefold()),
    }
    if timing_labels is not None:
        report["timing_labels"] = dict(stage_labels)
    write_outputs(
        {
            output_dir / "batch-summary.json": json.dumps(report, ensure_ascii=False, indent=2)
            + "\n",
            output_dir / "timing-report.txt": format_timing_report(report),
        }
    )
    return report


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="批量处理本地文件夹中的音频，生成发言人和时间戳")
    parser.add_argument("folder", type=Path, help="云桌面上的音频文件夹")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/batch"),
        help="输出父目录；每次新建带北京时间戳和唯一标识的批次子目录",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        help="续跑已有批次目录，自动复用已有有效 TXT/JSON",
    )
    parser.add_argument(
        "--log-file", type=Path, help="日志文件，默认输出目录下的 batch.log（追加）"
    )
    parser.add_argument("--recursive", action="store_true", help="同时处理子文件夹")
    parser.add_argument("--file-workers", type=int, default=1, help="同时处理文件数，默认 1")
    parser.add_argument(
        "--asr-workers", type=int, help="每个文件内部的 ASR 分块并发数，默认读取 ASR_WORKERS（1）"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="复用当前批次的 TXT/JSON；续跑请用 --resume-dir",
    )
    parser.add_argument(
        "--timeout-seconds", type=float, help="每次远程请求的读写等待超时秒数，默认读取 .env"
    )
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    args = parser.parse_args(argv)
    if args.file_workers < 1:
        parser.error("--file-workers 必须大于 0")
    if args.asr_workers is not None and args.asr_workers < 1:
        parser.error("--asr-workers 必须大于 0")
    try:
        if args.resume_dir is not None:
            args.output_dir = args.resume_dir.resolve()
            if not args.output_dir.is_dir():
                parser.error("--resume-dir 必须指向已有的批次目录")
            args.skip_existing = True
        else:
            args.output_dir = create_batch_directory(args.output_dir)
        log_path = (args.log_file or args.output_dir / "batch.log").resolve()
        with batch_logging(log_path):
            logger.info(
                "批次目录：%s（%s）", args.output_dir, "续跑" if args.resume_dir else "新建"
            )
            logger.info("批处理日志文件：%s", log_path)
            if args.skip_existing and args.resume_dir is None:
                logger.info(
                    "本次使用新批次目录；若要复用之前的结果，请用 --resume-dir 指定旧批次目录"
                )
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
        settings = replace(Settings.from_env(), job_workers=args.file_workers)
        if args.asr_workers is not None:
            settings = replace(settings, asr_workers=args.asr_workers)
        if args.timeout_seconds is not None:
            settings = replace(settings, api_timeout_seconds=args.timeout_seconds)
        if not settings.asr_api_url:
            raise ValueError("请先在 .env 中配置 ASR_API_URL")
        logger.info(
            "本次配置：%d 个文件并发，每个文件 %d 个 ASR 分块并发，"
            "ASR 请求合计最多 %d 个，ASR 子块上限 %.1f 秒，ASR 等待上限 %.1f 秒，"
            "说话人读写等待上限 %.1f 秒",
            settings.job_workers,
            settings.asr_workers,
            settings.job_workers * settings.asr_workers,
            settings.asr_chunk_seconds,
            min(settings.api_timeout_seconds, settings.asr_timeout_seconds),
            settings.api_timeout_seconds,
        )
        pipeline = MeetingPipeline(settings)
        report = run_batch(
            args.folder,
            args.output_dir,
            pipeline,
            workers=args.file_workers,
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
        f"结果目录：{args.output_dir.resolve()}\n"
        f"汇总：{args.output_dir.resolve() / 'batch-summary.json'}\n"
        f"耗时报告：{args.output_dir.resolve() / 'timing-report.txt'}\n"
        f"批次总耗时：{duration_text(report['processing_seconds'])}；"
        f"平均单文件耗时：{duration_text(report['timing_summary']['processed_files']['average_seconds'])}"
    )
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
