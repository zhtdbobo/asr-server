"""Start gpu_services:diarization_app with one worker for Community-1."""

import hmac
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from audio import AudioInputError, decode_audio
from config import Settings
from offline_docs import install_offline_docs
from pipeline import validate_options

logger = logging.getLogger(__name__)


def check_key(expected: str, authorization: str | None):
    if expected and not hmac.compare_digest(authorization or "", f"Bearer {expected}"):
        raise HTTPException(401, "无效的 API Key", headers={"WWW-Authenticate": "Bearer"})


def save_upload(file: UploadFile, target: Path, limit_mb: int):
    total = 0
    with target.open("wb") as stream:
        while chunk := file.file.read(1024 * 1024):
            total += len(chunk)
            if total > limit_mb * 1024 * 1024:
                raise HTTPException(413, f"文件超过 {limit_mb} MB 限制")
            stream.write(chunk)
    if total == 0:
        raise HTTPException(400, "上传文件为空")


class GPUEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.lock = Lock()

    @property
    def loaded(self):
        return self.model is not None

    def load(self):
        model_path = Path(self.settings.diarization_model).expanduser().resolve()
        config_name = "config.yaml"
        if not model_path.is_dir() or not (model_path / config_name).is_file():
            raise RuntimeError(
                f"DIARIZATION_MODEL 必须指向包含 {config_name} 的完整本地模型目录；"
                f"当前路径: {model_path}"
            )
        # Set before importing model libraries; this service only loads offline model files.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
        import torch

        device = self.settings.diarization_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("diarization 需要可用的 CUDA GPU，请检查驱动和容器 GPU 映射")
        from pyannote.audio import Pipeline

        model = Pipeline.from_pretrained(str(model_path))
        if model is None:
            raise RuntimeError("Community-1 加载失败，请检查本地模型目录完整性")
        model.to(torch.device(device))
        self.model = model

    def run(self, path, **options):
        import torch

        waveform, sr = decode_audio(path, self.settings.max_audio_seconds)
        with self.lock, torch.inference_mode():
            result = self.model(
                {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": sr},
                **options,
            )
            exclusive = getattr(result, "exclusive_speaker_diarization", None)
            if exclusive is None:
                raise RuntimeError("模型未返回 exclusive_speaker_diarization，请使用 Community-1")
            return {
                "turns": [
                    {"start": float(t.start), "end": float(t.end), "speaker": str(label)}
                    for t, _, label in exclusive.itertracks(yield_label=True)
                ]
            }


def create_gpu_app(engine=None, settings=None):
    settings = settings or Settings.from_env()
    engine = engine if engine is not None else GPUEngine(settings)
    pending = BoundedSemaphore(settings.gpu_pending_requests)

    @asynccontextmanager
    async def lifespan(_app):
        engine.load()
        yield

    app = FastAPI(title="diarization GPU service", lifespan=lifespan, docs_url=None, redoc_url=None)
    install_offline_docs(app)

    @app.get("/health")
    def health():
        return {"status": "ok", "models_loaded": engine.loaded, "service": "diarization"}

    def execute(file, authorization, **options):
        try:
            check_key(settings.internal_api_key, authorization)
            if not pending.acquire(blocking=False):
                raise HTTPException(503, "GPU 服务处理队列已满", headers={"Retry-After": "2"})
            try:
                with tempfile.TemporaryDirectory(prefix="diarization-") as directory:
                    path = Path(directory) / "input.audio"
                    save_upload(file, path, settings.max_upload_mb)
                    return engine.run(path, **options)
            finally:
                pending.release()
        except AudioInputError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("GPU inference failed")
            raise HTTPException(500, str(exc)) from exc
        finally:
            file.file.close()

    @app.post("/v1/diarization")
    def diarize(
        file: Annotated[UploadFile, File()],
        authorization: Annotated[str | None, Header()] = None,
        num_speakers: Annotated[int | None, Form(ge=1)] = None,
        min_speakers: Annotated[int | None, Form(ge=1)] = None,
        max_speakers: Annotated[int | None, Form(ge=1)] = None,
    ):
        try:
            _, options = validate_options("Chinese", num_speakers, min_speakers, max_speakers)
        except ValueError as exc:
            file.file.close()
            raise HTTPException(422, str(exc)) from exc
        return execute(file, authorization, **options)

    return app


diarization_app = create_gpu_app()
