"""Start gpu_services:diarization_app OR gpu_services:aligner_app, one worker each."""

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
    def __init__(self, kind: str, settings: Settings):
        self.kind, self.settings = kind, settings
        self.model = None
        self.lock = Lock()

    @property
    def loaded(self):
        return self.model is not None

    def load(self):
        model_path = Path(getattr(self.settings, f"{self.kind}_model")).expanduser().resolve()
        config_name = "config.yaml" if self.kind == "diarization" else "config.json"
        if not model_path.is_dir() or not (model_path / config_name).is_file():
            raise RuntimeError(
                f"{self.kind.upper()}_MODEL 必须指向包含 {config_name} 的完整本地模型目录；"
                f"当前路径: {model_path}"
            )
        # Set before importing model libraries; this service only loads offline model files.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
        import torch

        device = getattr(self.settings, f"{self.kind}_device")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"{self.kind} 需要可用的 CUDA GPU，请检查驱动和容器 GPU 映射")
        if self.kind == "diarization":
            from pyannote.audio import Pipeline

            model = Pipeline.from_pretrained(str(model_path))
            if model is None:
                raise RuntimeError("Community-1 加载失败，请检查本地模型目录完整性")
            model.to(torch.device(device))
        else:
            from qwen_asr import Qwen3ForcedAligner

            dtype = torch.float32
            if device.startswith("cuda"):
                with torch.cuda.device(device):
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model = Qwen3ForcedAligner.from_pretrained(
                str(model_path),
                device_map=device,
                dtype=dtype,
                attn_implementation="sdpa",
                local_files_only=True,
            )
        self.model = model

    def run(self, path, **options):
        import torch

        limit = self.settings.max_audio_seconds if self.kind == "diarization" else 180
        waveform, sr = decode_audio(path, limit)
        with self.lock, torch.inference_mode():
            if self.kind == "diarization":
                result = self.model(
                    {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": sr},
                    **options,
                )
                exclusive = getattr(result, "exclusive_speaker_diarization", None)
                if exclusive is None:
                    raise RuntimeError(
                        "模型未返回 exclusive_speaker_diarization，请使用 Community-1"
                    )
                return {
                    "turns": [
                        {"start": float(t.start), "end": float(t.end), "speaker": str(label)}
                        for t, _, label in exclusive.itertracks(yield_label=True)
                    ]
                }
            result = self.model.align(audio=(waveform, sr), **options)[0]
            return {
                "words": [
                    {"text": str(w.text), "start": float(w.start_time), "end": float(w.end_time)}
                    for w in result
                ]
            }


def create_gpu_app(kind: str, engine=None, settings=None):
    settings = settings or Settings.from_env()
    engine = engine if engine is not None else GPUEngine(kind, settings)
    pending = BoundedSemaphore(settings.gpu_pending_requests)

    @asynccontextmanager
    async def lifespan(_app):
        engine.load()
        yield

    app = FastAPI(title=f"{kind} GPU service", lifespan=lifespan, docs_url=None, redoc_url=None)
    install_offline_docs(app)

    @app.get("/health")
    def health():
        return {"status": "ok", "models_loaded": engine.loaded, "service": kind}

    def execute(file, authorization, **options):
        try:
            check_key(settings.internal_api_key, authorization)
            if not pending.acquire(blocking=False):
                raise HTTPException(503, "GPU 服务处理队列已满", headers={"Retry-After": "2"})
            try:
                with tempfile.TemporaryDirectory(prefix=f"{kind}-") as directory:
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

    if kind == "diarization":

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
    else:

        @app.post("/v1/alignment")
        def align(
            file: Annotated[UploadFile, File()],
            text: Annotated[str, Form(min_length=1)],
            authorization: Annotated[str | None, Header()] = None,
            language: Annotated[str, Form()] = "Chinese",
        ):
            try:
                language, _ = validate_options(language)
            except ValueError as exc:
                file.file.close()
                raise HTTPException(422, str(exc)) from exc
            return execute(file, authorization, text=text, language=language)

    return app


diarization_app = create_gpu_app("diarization")
aligner_app = create_gpu_app("aligner")
