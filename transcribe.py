import argparse
import json
import logging
import sys
from pathlib import Path

from pipeline import MeetingPipeline


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="生成带发言人和时间戳的会议转写")
    parser.add_argument("audio", type=Path, help="录音或视频文件路径")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--num-speakers", type=int, help="确定人数时指定，如 3")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        result = MeetingPipeline().transcribe(
            args.audio,
            language=args.language,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
    except (ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"处理失败: {exc}\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.audio.stem
    text_path = args.output_dir / f"{stem}.txt"
    json_path = args.output_dir / f"{stem}.json"
    text_path.write_text(result["formatted_text"] + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result["formatted_text"])
    print(f"\n已保存: {text_path.resolve()}\n        {json_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
