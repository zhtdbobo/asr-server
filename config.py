import math
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    asr_api_url: str = ""
    asr_model: str = ""
    asr_api_key: str = field(default="", repr=False)
    diarization_api_url: str = "http://diarization:8002/v1/diarization"
    internal_api_key: str = field(default="", repr=False)
    service_api_key: str = field(default="", repr=False)
    api_timeout_seconds: float = 600
    asr_timeout_seconds: float = 90
    asr_max_completion_tokens: int = 1024
    asr_chunk_seconds: float = 30
    asr_workers: int = 1
    diarization_model: str = "/models/speaker-diarization-community-1"
    diarization_device: str = "cuda:0"
    gpu_pending_requests: int = 4
    job_workers: int = 2
    max_pending_jobs: int = 100
    job_directory: str = "outputs/jobs"
    merge_gap_seconds: float = 1.5
    max_upload_mb: int = 500
    max_audio_seconds: float = 3600
    firered_asr_api_url: str = ""
    firered_asr_model: str = ""
    firered_asr_api_key: str = field(default="", repr=False)
    nemotron_diarization_api_url: str = ""
    nemotron_api_key: str = field(default="", repr=False)
    default_backend: str = "firered-nemotron"

    def __post_init__(self):
        if self.default_backend not in ("qwen-community1", "firered-nemotron"):
            raise ValueError("DEFAULT_BACKEND 必须为 qwen-community1 或 firered-nemotron")
        if not math.isfinite(self.merge_gap_seconds) or self.merge_gap_seconds < 0:
            raise ValueError("merge_gap_seconds 必须是大于等于 0 的有限数字")
        for name in (
            "max_audio_seconds",
            "api_timeout_seconds",
            "asr_timeout_seconds",
            "asr_chunk_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if (
            isinstance(self.asr_max_completion_tokens, bool)
            or not isinstance(self.asr_max_completion_tokens, int)
            or self.asr_max_completion_tokens <= 0
        ):
            raise ValueError("asr_max_completion_tokens 必须为正整数")
        if (
            isinstance(self.asr_workers, bool)
            or not isinstance(self.asr_workers, int)
            or self.asr_workers <= 0
        ):
            raise ValueError("asr_workers 必须为正整数")
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
