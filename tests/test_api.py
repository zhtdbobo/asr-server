from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import create_app
from audio import AudioInputError
from config import Settings
from pipeline import PipelineBusyError


class FakePipeline:
    loaded = False

    def __init__(self):
        self.calls = []
        self.error = None

    def transcribe(self, path, **options):
        self.calls.append((Path(path), options, Path(path).read_bytes()))
        if self.error:
            raise self.error
        return {"formatted_text": "发言人1\n00:00\n你好。", "segments": []}


@pytest.fixture
def endpoint(tmp_path):
    pipeline = FakePipeline()
    with TestClient(
        create_app(
            pipeline,
            Settings(
                max_upload_mb=1, job_directory=str(tmp_path), default_backend="qwen-community1"
            ),
        )
    ) as client:
        yield client, pipeline


def test_json_upload_and_cleanup(endpoint):
    client, pipeline = endpoint
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("meeting.wav", b"audio-data")},
        data={"num_speakers": "3"},
    )
    assert response.status_code == 200
    assert response.json()["formatted_text"].startswith("发言人1")
    path, options, body = pipeline.calls[0]
    assert options["num_speakers"] == 3
    assert body == b"audio-data"
    assert not path.exists()


def test_backend_selection(tmp_path, monkeypatch):
    qwen, firered = FakePipeline(), FakePipeline()
    monkeypatch.setattr("app.MeetingPipeline", lambda *args, **kwargs: firered)
    monkeypatch.setattr("app.FireRedNemotronClients", lambda settings: object())
    settings = Settings(
        job_directory=str(tmp_path),
        firered_asr_api_url="http://firered/transcriptions",
        nemotron_diarization_api_url="http://nemotron/diarization",
    )
    with TestClient(create_app(qwen, settings)) as client:
        files = {"file": ("meeting.wav", b"audio")}
        url = "/v1/audio/transcriptions"
        assert client.post(url, files=files).status_code == 200
        assert client.post(url, files=files, data={"backend": "qwen-community1"}).status_code == 200
        assert client.post(url, files=files, data={"backend": "unknown"}).status_code == 422
        response = client.post("/v1/jobs", files=files, data={"backend": "qwen-community1"})
        assert response.status_code == 202
        with client.app.state.jobs.connect() as db:
            options = db.execute("SELECT options FROM jobs").fetchone()[0]
        assert '"backend": "qwen-community1"' in options
        assert len(firered.calls) == 1
        assert qwen.calls


def test_text_response(endpoint):
    client, _ = endpoint
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("meeting.m4a", b"audio")},
        data={"response_format": "text"},
    )
    assert response.status_code == 200
    assert response.text == "发言人1\n00:00\n你好。"
    assert "text/plain" in response.headers["content-type"]


@pytest.mark.parametrize(
    "data",
    [
        {"num_speakers": "3", "min_speakers": "2"},
        {"min_speakers": "4", "max_speakers": "2"},
        {"num_speakers": "0"},
        {"language": "unknown"},
        {"response_format": "srt"},
    ],
)
def test_invalid_options_rejected_before_inference(endpoint, data):
    client, pipeline = endpoint
    response = client.post(
        "/v1/audio/transcriptions", files={"file": ("a.wav", b"data")}, data=data
    )
    assert response.status_code == 422
    assert pipeline.calls == []


def test_empty_and_oversized_uploads(endpoint):
    client, pipeline = endpoint
    for body, expected in [(b"", 400), (b"a" * (1024 * 1024 + 1), 413)]:
        response = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", body)})
        assert response.status_code == expected
    assert pipeline.calls == []


@pytest.mark.parametrize(
    "error,status",
    [
        (PipelineBusyError("busy"), 503),
        (AudioInputError("invalid audio"), 400),
        (RuntimeError("models failed"), 500),
    ],
)
def test_failed_requests_cleanup_and_status(endpoint, error, status):
    client, pipeline = endpoint
    pipeline.error = error
    response = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"data")})
    assert response.status_code == status
    assert not pipeline.calls[0][0].exists()
    if status == 503:
        assert response.headers["retry-after"] == "5"


def test_health_and_openapi_do_not_load_models(endpoint):
    client, pipeline = endpoint
    assert client.get("/health").json() == {"status": "ok", "mode": "remote", "job_workers": 2}
    assert "/v1/audio/transcriptions" in client.get("/openapi.json").json()["paths"]
    assert pipeline.calls == []
