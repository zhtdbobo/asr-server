"""Transcribe a local audio folder using the three remote inference services."""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from config import Settings
from pipeline import MeetingPipeline, validate_options

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

    def process(path):
        relative = path.relative_to(folder)
        # Retain the extension so meeting.wav and meeting.mp3 cannot overwrite each other.
        text_path = output_dir / relative.parent / f"{relative.name}.txt"
        json_path = output_dir / relative.parent / f"{relative.name}.json"
        item = {
            "file": relative.as_posix(),
            "text_file": str(text_path),
            "json_file": str(json_path),
        }
        if skip_existing and text_path.is_file() and json_path.is_file():
            return {**item, "status": "skipped"}
        try:
            result = pipeline.transcribe(path, **options)
            write_outputs(
                {
                    text_path: result["formatted_text"] + "\n",
                    json_path: json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                }
            )
            return {**item, "status": "completed"}
        except Exception as exc:  # noqa: BLE001 - A failed input must not abort other files.
            return {**item, "status": "failed", "error": str(exc)}

    results = []
    labels = {"completed": "成功", "failed": "失败", "skipped": "跳过"}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(process, path) for path in files]
        for future in as_completed(pending):
            item = future.result()
            results.append(item)
            suffix = f"：{item['error']}" if item["status"] == "failed" else ""
            print(
                f"[{len(results)}/{len(files)}] {labels[item['status']]} {item['file']}{suffix}",
                flush=True,
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
    parser.add_argument("--recursive", action="store_true", help="同时处理子文件夹")
    parser.add_argument("--workers", type=int, default=1, help="同时处理文件数，默认 1")
    parser.add_argument("--skip-existing", action="store_true", help="TXT、JSON 均已存在时跳过")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers 必须大于 0")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pipeline = None
    try:
        language, counts = validate_options(
            args.language, args.num_speakers, args.min_speakers, args.max_speakers
        )
        settings = replace(Settings.from_env(), job_workers=args.workers)
        if not settings.asr_api_url:
            raise ValueError("请先在 .env 中配置 ASR_API_URL")
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
        parser.exit(1, f"批处理失败: {exc}\n")
    finally:
        if pipeline is not None:
            pipeline.close()
    print(
        f"\n处理结束：共 {report['total']} 个，成功 {report['completed']} 个，"
        f"失败 {report['failed']} 个，跳过 {report['skipped']} 个。\n"
        f"汇总：{args.output_dir.resolve() / 'batch-summary.json'}"
    )
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
