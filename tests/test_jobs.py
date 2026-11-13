import io
import sqlite3
import time
from threading import Event

import pytest
from fastapi.testclient import TestClient

from app import create_app
from config import Settings
from jobs import JobManager


def wait_status(client, job_id, expected):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = client.get(f"/v1/jobs/{job_id}").json()
        if status["status"] == expected:
            return status
        time.sleep(0.02)
    pytest.fail(f"job never reached {expected}: {status}")


class ControlledPipeline:
    def __init__(self):
        self.entered, self.release = Event(), Event()
        self.error = None

    def transcribe(self, path, **kwargs):
        assert path.read_bytes() == b"audio"
        self.entered.set()
        assert self.release.wait(5)
        if self.error:
            raise self.error
        return {"formatted_text": "发言人1\n00:00\n你好。", "segments": []}


def test_queued_job_lifecycle_backpressure_and_auth(tmp_path):
    engine = ControlledPipeline()
    settings = Settings(
        job_directory=str(tmp_path),
        job_workers=1,
        max_pending_jobs=1,
        service_api_key="public-key",
        default_backend="qwen-community1",
    )
    with TestClient(create_app(engine, settings)) as client:
        assert client.post("/v1/jobs", files={"file": ("a.wav", b"audio")}).status_code == 401
        client.headers["Authorization"] = "Bearer public-key"
        response = client.post("/v1/jobs", files={"file": ("a.wav", b"audio")})
        assert response.status_code == 202
        job_id = response.json()["id"]
        assert engine.entered.wait(3)
        assert client.get(f"/v1/jobs/{job_id}/result").status_code == 202
        assert client.delete(f"/v1/jobs/{job_id}").status_code == 409
        assert client.post("/v1/jobs", files={"file": ("a.wav", b"audio")}).status_code == 429
        engine.release.set()
        wait_status(client, job_id, "completed")
        response = client.get(f"/v1/jobs/{job_id}/result?response_format=text")
        assert response.text == "发言人1\n00:00\n你好。"
        assert client.get(f"/v1/jobs/{job_id}/result").json()["segments"] == []
        assert client.delete(f"/v1/jobs/{job_id}").status_code == 204
        assert client.get(f"/v1/jobs/{job_id}").status_code == 404
        assert list(tmp_path.glob("*.audio")) == []


def test_failed_job_and_bad_upload_cleanup(tmp_path):
    engine = ControlledPipeline()
    engine.error = RuntimeError("upstream unavailable")
    engine.release.set()
    settings = Settings(job_directory=str(tmp_path), default_backend="qwen-community1")
    with TestClient(create_app(engine, settings)) as client:
        assert client.post("/v1/jobs", files={"file": ("empty.wav", b"")}).status_code == 400
        job_id = client.post("/v1/jobs", files={"file": ("a.wav", b"audio")}).json()["id"]
        status = wait_status(client, job_id, "failed")
        assert status["error"] == "upstream unavailable"
        assert client.get(f"/v1/jobs/{job_id}/result").status_code == 409
        assert client.get("/v1/jobs/invalid-id").status_code == 404


def test_restart_requeues_interrupted_job(tmp_path):
    engine = ControlledPipeline()
    engine.release.set()
    settings = Settings(job_directory=str(tmp_path), job_workers=1)
    first = JobManager(engine, settings)
    first.start()
    assert first.stop()
    job_id = first.submit(io.BytesIO(b"audio"), {"language": "Chinese"})
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as db:
        db.execute("UPDATE jobs SET status='running' WHERE id=?", (job_id,))
    with TestClient(create_app(engine, settings)) as client:
        wait_status(client, job_id, "completed")
        assert client.get(f"/v1/jobs/{job_id}/result").status_code == 200
