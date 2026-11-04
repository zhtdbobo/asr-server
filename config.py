import math
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    asr_api_url: str = ""
    asr_model: str = ""
    asr_api_key: str = field(default="", repr=False)
    diarization_api_url: str = "http://diarization:8002/v1/diarization"
    aligner_api_url: str = "http://aligner:8003/v1/alignment"
    internal_api_key: str = field(default="", repr=False)
    service_api_key: str = field(default="", repr=False)
    api_timeout_seconds: float = 600
    asr_chunk_seconds: float = 30
    aligner_model: str = "/models/Qwen3-ForcedAligner-0.6B"
    diarization_model: str = "/models/speaker-diarization-community-1"
    aligner_device: str = "cuda:0"
    diarization_device: str = "cuda:0"
    gpu_pending_requests: int = 4
    job_workers: int = 2
    max_pending_jobs: int = 100
    job_directory: str = "outputs/jobs"
    merge_gap_seconds: float = 1.5
    max_speaker_distance: float = 0.75
    max_upload_mb: int = 500
    max_audio_seconds: float = 3600

    def __post_init__(self):
        for name in ("merge_gap_seconds", "max_speaker_distance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是大于等于 0 的有限数字")
        for name in ("max_audio_seconds", "api_timeout_seconds", "asr_chunk_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.asr_chunk_seconds > 180:
            raise ValueError("asr_chunk_seconds 不能超过对齐模型的 180 秒分块上限")
        for name in ("max_upload_mb", "gpu_pending_requests", "job_workers", "max_pending_jobs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须大于 0")

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv

        load_dotenv()
        defaults = cls()
        return cls(
            **{
                name: type(getattr(defaults, name))(
                    os.environ.get(name.upper(), getattr(defaults, name))
                )
                for name in cls.__dataclass_fields__
            }
        )
