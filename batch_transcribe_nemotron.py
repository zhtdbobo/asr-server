"""FireRedASR2(ASR) + Nemotron-3-Diarization 批处理入口。

复用 batch_transcribe.py 的全部批处理/报告逻辑，仅注入 FireRedNemotronClients：
  - ASR：FireRedASR2（普通 JSON，带标点）           → ASR_API_URL
  - 说话人分离：Nemotron-3-Diarization（segments）   → DIARIZATION_API_URL

优先加载同目录 .env.nemotron（存在时 override）。
用法同 batch_transcribe.py：
  python batch_transcribe_nemotron.py <音频目录> --output-dir outputs/xxx --file-workers 1
"""

import sys
from dataclasses import replace
from pathlib import Path

_env = Path(__file__).resolve().parent / ".env.nemotron"
if _env.is_file():
    from dotenv import load_dotenv

    load_dotenv(_env, override=True)

import batch_transcribe as base
from config import Settings
from firered_nemotron_client import FireRedNemotronClients
from pipeline import MeetingPipeline, validate_options


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
            raise ValueError("请先在 .env.nemotron 中配置 ASR_API_URL（FireRedASR2 服务）")
        base.logger.info(
            "本次配置（FireRedASR2 + Nemotron-Diarization）：%d 文件并发，每文件 %d ASR 分块，ASR=%s，DIAR=%s",
            settings.job_workers,
            settings.asr_workers,
            settings.asr_api_url,
            settings.diarization_api_url,
        )
        pipeline = MeetingPipeline(settings, clients=FireRedNemotronClients(settings))
        report = base.run_batch(
            args.folder,
            args.output_dir,
            pipeline,
            workers=args.file_workers,
            recursive=args.recursive,
            skip_existing=args.skip_existing,
            options={"language": language, **counts},
        )
    except (ValueError, RuntimeError, OSError) as exc:
        base.logger.error("批处理退出（%s）：%s", type(exc).__name__, exc)
        parser.exit(1, f"批处理失败: {exc}\n")
    finally:
        if pipeline is not None:
            pipeline.close()
    base.logger.info(
        f"\n处理结束：共 {report['total']} 个，成功 {report['completed']} 个，"
        f"失败 {report['failed']} 个，跳过 {report['skipped']} 个。\n"
        f"结果目录：{args.output_dir.resolve()}\n"
        f"批次总耗时：{base.duration_text(report['processing_seconds'])}"
    )
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    base.run_cli = run_cli
    sys.exit(base.main())
