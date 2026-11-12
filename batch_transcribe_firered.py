"""FireRedASR2 版本批处理入口。

复用 batch_transcribe.py 里的全部批处理/报告逻辑（run_batch、报告、参数解析），
仅把 ASR 客户端替换为 FireRedServiceClients（FireRedASR2 普通 JSON 接口）。
说话人分离仍走 Community-1（DIARIZATION_API_URL）。

用法与 batch_transcribe.py 完全一致，例如：
  python batch_transcribe_firered.py D:\\audio --output-dir outputs/batch --file-workers 2
需要在本目录提供 .env（或 .env.firered），至少配置：
  ASR_API_URL=http://127.0.0.1:8004/v1/audio/transcriptions
  ASR_MODEL=fireredasr2
  DIARIZATION_API_URL=http://127.0.0.1:8002/v1/diarization
"""

import sys
from dataclasses import replace
from pathlib import Path

# 优先加载同目录 .env.firered（存在时覆盖 .env 的对应变量）
_env_firered = Path(__file__).resolve().parent / ".env.firered"
if _env_firered.is_file():
    from dotenv import load_dotenv

    load_dotenv(_env_firered, override=True)

import batch_transcribe as base
from config import Settings
from firered_client import FireRedServiceClients
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
            raise ValueError("请先在 .env 中配置 ASR_API_URL（FireRedASR2 服务地址）")
        base.logger.info(
            "本次配置（FireRedASR2）：%d 个文件并发，每个文件 %d 个 ASR 分块并发，"
            "ASR 子块上限 %.1f 秒，ASR 地址 %s",
            settings.job_workers,
            settings.asr_workers,
            settings.asr_chunk_seconds,
            settings.asr_api_url,
        )
        # 注入 FireRed 客户端；pipeline 的其余逻辑不变
        pipeline = MeetingPipeline(settings, clients=FireRedServiceClients(settings))
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
        f"汇总：{args.output_dir.resolve() / 'batch-summary.json'}\n"
        f"耗时报告：{args.output_dir.resolve() / 'timing-report.txt'}\n"
        f"批次总耗时：{base.duration_text(report['processing_seconds'])}；"
        f"平均单文件耗时：{base.duration_text(report['timing_summary']['processed_files']['average_seconds'])}"
    )
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    # 复用原脚本的 main()，只是把 run_cli 换成上面的实现（base.run_cli 被本模块覆盖）
    base.run_cli = run_cli
    sys.exit(base.main())
