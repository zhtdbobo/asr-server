import logging
import tempfile
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from threading import BoundedSemaphore
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from audio import AudioInputError
from clients import UpstreamError
from config import Settings
from firered_nemotron_client import FireRedNemotronClients
from gpu_services import check_key, save_upload
from jobs import JobManager, QueueFullError
from offline_docs import install_offline_docs
from pipeline import MeetingPipeline, PipelineBusyError, validate_options

logger = logging.getLogger(__name__)
Backend = Literal["qwen-community1", "firered-nemotron"]


class TranscriptionBackends:
    def __init__(self, settings: Settings, default_pipeline):
        slots = BoundedSemaphore(settings.job_workers)
        self.default = (
            default_pipeline
            if default_pipeline is not None
            else MeetingPipeline(settings, slots=slots)
        )
        self.firered = None
        if settings.firered_asr_api_url and settings.nemotron_diarization_api_url:
            alternate = replace(
                settings,
                asr_api_url=settings.firered_asr_api_url,
                asr_model=settings.firered_asr_model,
                asr_api_key=settings.firered_asr_api_key,
                diarization_api_url=settings.nemotron_diarization_api_url,
                internal_api_key=settings.nemotron_api_key,
            )
            self.firered = MeetingPipeline(
                alternate, clients=FireRedNemotronClients(alternate), slots=slots
            )

    def select(self, backend: Backend):
        if backend == "qwen-community1":
            return self.default
        if backend == "firered-nemotron" and self.firered is not None:
            return self.firered
        raise HTTPException(503, f"转写后端未配置：{backend}")

    def transcribe(self, path, *, backend: Backend = "qwen-community1", **options):
        return self.select(backend).transcribe(path, **options)

    def close(self):
        for pipeline in (self.default, self.firered):
            if hasattr(pipeline, "close"):
                pipeline.close()


def create_app(pipeline=None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    backends = TranscriptionBackends(settings, pipeline)
    jobs = JobManager(backends, settings)

    @asynccontextmanager
    async def lifespan(_app):
        jobs.start()
        try:
            yield
        finally:
            if jobs.stop():
                backends.close()

    app = FastAPI(
        title="会议转写任务服务",
        version="0.2.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    install_offline_docs(app)
    app.state.pipeline, app.state.jobs = backends.default, jobs

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": "remote", "job_workers": settings.job_workers}

    def options(language, num_speakers, min_speakers, max_speakers):
        try:
            language, counts = validate_options(language, num_speakers, min_speakers, max_speakers)
            return {"language": language, **counts}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/v1/audio/transcriptions", response_model=None)
    def transcribe(
        file: Annotated[UploadFile, File(description="完整录音文件")],
        authorization: Annotated[str | None, Header()] = None,
        language: Annotated[str, Form()] = "Chinese",
        num_speakers: Annotated[int | None, Form(ge=1)] = None,
        min_speakers: Annotated[int | None, Form(ge=1)] = None,
        max_speakers: Annotated[int | None, Form(ge=1)] = None,
        response_format: Annotated[Literal["json", "text"], Form()] = "json",
        backend: Annotated[Backend, Form()] = settings.default_backend,
    ) -> Response:
        try:
            check_key(settings.service_api_key, authorization)
            selected_pipeline = backends.select(backend)
            kwargs = options(language, num_speakers, min_speakers, max_speakers)
            with tempfile.TemporaryDirectory(prefix="meeting-upload-") as directory:
                path = Path(directory) / "input.audio"
                save_upload(file, path, settings.max_upload_mb)
                output = selected_pipeline.transcribe(path, **kwargs)
            return (
                PlainTextResponse(output["formatted_text"])
                if response_format == "text"
                else JSONResponse(output)
            )
        except PipelineBusyError as exc:
            raise HTTPException(503, str(exc), headers={"Retry-After": "5"}) from exc
        except AudioInputError as exc:
            raise HTTPException(400, str(exc)) from exc
        except UpstreamError as exc:
            raise HTTPException(502, str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("Transcription failed")
            raise HTTPException(500, str(exc)) from exc
        finally:
            file.file.close()

    @app.post("/v1/jobs", status_code=202)
    def submit(
        file: Annotated[UploadFile, File()],
        authorization: Annotated[str | None, Header()] = None,
        language: Annotated[str, Form()] = "Chinese",
        num_speakers: Annotated[int | None, Form(ge=1)] = None,
        min_speakers: Annotated[int | None, Form(ge=1)] = None,
        max_speakers: Annotated[int | None, Form(ge=1)] = None,
        backend: Annotated[Backend, Form()] = settings.default_backend,
    ):
        try:
            check_key(settings.service_api_key, authorization)
            backends.select(backend)
            kwargs = options(language, num_speakers, min_speakers, max_speakers)
            job_id = jobs.submit(file.file, {"backend": backend, **kwargs})
            return {
                "id": job_id,
                "status_url": f"/v1/jobs/{job_id}",
                "result_url": f"/v1/jobs/{job_id}/result",
            }
        except QueueFullError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "10"}) from exc
        finally:
            file.file.close()

    @app.get("/v1/jobs/{job_id}")
    def status(job_id: str, authorization: Annotated[str | None, Header()] = None):
        check_key(settings.service_api_key, authorization)
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "任务不存在") from exc

    @app.get("/v1/jobs/{job_id}/result", response_model=None)
    def result(
        job_id: str,
        response_format: Literal["json", "text"] = "json",
        authorization: Annotated[str | None, Header()] = None,
    ):
        check_key(settings.service_api_key, authorization)
        try:
            state, output = jobs.result(job_id)
        except KeyError as exc:
            raise HTTPException(404, "任务不存在") from exc
        if output is None:
            return JSONResponse(state, status_code=409 if state["status"] == "failed" else 202)
        return (
            PlainTextResponse(output["formatted_text"])
            if response_format == "text"
            else JSONResponse(output)
        )

    @app.delete("/v1/jobs/{job_id}", status_code=204)
    def delete(job_id: str, authorization: Annotated[str | None, Header()] = None):
        check_key(settings.service_api_key, authorization)
        try:
            jobs.delete(job_id)
        except KeyError as exc:
            raise HTTPException(404, "任务不存在") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return Response(status_code=204)

    return app


app = create_app()
