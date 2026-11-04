import io
import json
from email import policy
from email.parser import BytesParser

import httpx
import numpy as np
import pytest
import soundfile as sf

from clients import ServiceClients, UpstreamError
from config import Settings


def multipart(request):
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\nMIME-Version: 1.0\r\n\r\n".encode()
        + request.content
    )
    return {
        part.get_param("name", header="content-disposition"): (
            part.get_filename(),
            part.get_payload(decode=True),
        )
        for part in message.iter_parts()
    }


def test_file_upload_api_contract_and_auth():
    def handle(request):
        assert str(request.url) == "http://asr:8000/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer secret"
        data = multipart(request)
        assert data["file"][0] == "audio.wav"
        audio, sr = sf.read(io.BytesIO(data["file"][1]))
        assert sr == 16000 and len(audio) == 8000
        assert data["model"][1] == b"Qwen/Qwen3-ASR-1.7B"
        assert data["response_format"][1] == b"json"
        assert data["stream"][1] == b"false"
        return httpx.Response(200, json={"text": "language Chinese<asr_text>你好。"})

    settings = Settings(
        asr_api_url="http://asr:8000/v1/audio/transcriptions",
        asr_model="Qwen/Qwen3-ASR-1.7B",
        asr_api_key="secret",
    )
    client = ServiceClients(settings, httpx.MockTransport(handle))
    try:
        assert client.recognize(np.zeros(8000, dtype=np.float32), 16000) == "你好。"
    finally:
        client.close()
    assert "secret" not in repr(settings)


def test_internal_service_payloads_and_reopened_retry(tmp_path, monkeypatch):
    import clients as module

    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    path = tmp_path / "original.audio"
    path.write_bytes(b"original-content")
    calls = []

    def handle(request):
        data = multipart(request)
        calls.append(data)
        assert request.headers["authorization"] == "Bearer internal"
        if request.url.path == "/v1/diarization":
            assert data["file"][1] == b"original-content"
            assert data["num_speakers"][1] == b"3"
            return (
                httpx.Response(503) if len(calls) == 1 else httpx.Response(200, json={"turns": []})
            )
        assert data["text"][1].decode() == "你好"
        assert data["language"][1] == b"Chinese"
        return httpx.Response(200, json={"words": []})

    client = ServiceClients(Settings(internal_api_key="internal"), httpx.MockTransport(handle))
    try:
        assert client.diarize(path, {"num_speakers": 3}) == []
        assert client.align(np.zeros(8000), 16000, "你好", "Chinese") == []
        assert len(calls) == 3
    finally:
        client.close()


@pytest.mark.parametrize(
    "status,body",
    [
        (401, {}),
        (200, {"bad": "missing text"}),
        (200, ["not a dict"]),
        (200, {"text": "partial", "finish_reason": "length"}),
    ],
)
def test_bad_upstream_response(status, body):
    client = ServiceClients(
        Settings(asr_api_url="http://asr/upload", asr_model="explicit-model"),
        httpx.MockTransport(lambda _: httpx.Response(status, content=json.dumps(body))),
    )
    try:
        with pytest.raises(UpstreamError):
            client.recognize(np.zeros(10), 16000)
    finally:
        client.close()


def test_discover_local_served_model_and_cache():
    calls = []
    model_id = "/data/shared/Qwen3-ASR/models/Qwen3-ASR-1.7B"

    def handle(request):
        calls.append(request.method)
        assert request.headers["authorization"] == "Bearer secret"
        if request.method == "GET":
            assert str(request.url) == "http://asr/prefix/v1/models"
            return httpx.Response(200, json={"data": [{"id": model_id}]})
        assert multipart(request)["model"][1].decode() == model_id
        return httpx.Response(200, json={"text": "你好"})

    client = ServiceClients(
        Settings(asr_api_url="http://asr/prefix/v1/audio/transcriptions", asr_api_key="secret"),
        httpx.MockTransport(handle),
    )
    try:
        assert client.recognize(np.zeros(10), 16000) == "你好"
        assert client.recognize(np.zeros(10), 16000) == "你好"
    finally:
        client.close()
    assert calls == ["GET", "POST", "POST"]


@pytest.mark.parametrize("entries", [[], [{"id": "one"}, {"id": "two"}], None])
def test_discovery_requires_one_model(entries):
    client = ServiceClients(
        Settings(asr_api_url="http://asr/v1/audio/transcriptions"),
        httpx.MockTransport(lambda _: httpx.Response(200, json={"data": entries})),
    )
    try:
        with pytest.raises(UpstreamError, match="ASR_MODEL"):
            client.recognize(np.zeros(10), 16000)
    finally:
        client.close()


def test_asr_404_reports_model_url_and_upstream_reason():
    client = ServiceClients(
        Settings(
            asr_api_url="http://user:password@asr/v1/audio/transcriptions?key=hidden",
            asr_model="wrong-model",
            asr_api_key="secret",
        ),
        httpx.MockTransport(
            lambda _: httpx.Response(
                404, json={"error": {"message": "The model wrong-model does not exist; secret"}}
            )
        ),
    )
    try:
        with pytest.raises(UpstreamError) as caught:
            client.recognize(np.zeros(10), 16000)
        message = str(caught.value)
        assert "Qwen3-ASR" in message and "HTTP 404" in message
        assert "http://asr/v1/audio/transcriptions" in message
        assert "wrong-model does not exist" in message and "ASR_MODEL" in message
        assert all(secret not in message for secret in ("secret", "password", "hidden"))
    finally:
        client.close()
