import io
import json
from email import policy
from email.parser import BytesParser

import httpx
import numpy as np
import pytest
import soundfile as sf

from clients import ChunkInferenceError, ServiceClients, UpstreamError
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


def stream_event(content="", finish=None):
    payload = {"choices": [{"delta": {"content": content}, "finish_reason": finish}]}
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()


def asr_response(text, finish="stop"):
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=stream_event(text) + stream_event(finish=finish) + b"data: [DONE]\n\n",
    )


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
        assert data["stream"][1] == b"true"
        assert data["max_completion_tokens"][1] == b"1024"
        assert data["temperature"][1] == b"0"
        assert request.extensions["timeout"]["read"] == 90
        return asr_response("language Chinese<asr_text>你好。")

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
        assert request.url.path == "/v1/diarization"
        assert data["file"][1] == b"original-content"
        assert data["num_speakers"][1] == b"3"
        return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, json={"turns": []})

    client = ServiceClients(Settings(internal_api_key="internal"), httpx.MockTransport(handle))
    try:
        assert client.diarize(path, {"num_speakers": 3}) == []
        assert len(calls) == 2
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
        return asr_response("你好")

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


def test_asr_timeout_does_not_resubmit_generation_without_logging_secrets(monkeypatch, caplog):
    import clients as module

    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    caplog.set_level("INFO", logger="clients")
    requests = []

    def timeout(request):
        requests.append(request)
        raise httpx.ReadTimeout("secret in transport details", request=request)

    client = ServiceClients(
        Settings(
            asr_api_url="http://asr/upload", asr_model="explicit-model", api_timeout_seconds=90
        ),
        httpx.MockTransport(timeout),
    )
    try:
        with pytest.raises(UpstreamError, match="未自动重复提交") as caught:
            client.recognize(np.zeros(10), 16000)
        assert len(requests) == 1
        assert "90 秒" in str(caught.value)
        messages = "\n".join(caplog.messages)
        assert "ReadTimeout" in messages and "不自动重复提交" in messages
        assert "secret" not in messages
    finally:
        client.close()


class ObservedStream(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.consumed = 0
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk

    def close(self):
        self.closed = True


def test_diarization_logs_headers_before_body_and_json_without_trace_secrets(tmp_path, caplog):
    caplog.set_level("INFO", logger="clients")
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")

    class DelayedBody(httpx.SyncByteStream):
        def __iter__(self):
            assert "收到响应头，HTTP 200" in caplog.text
            assert "JSON 解析完成" not in caplog.text
            yield b'{"turns": []}'

    def handle(request):
        request.extensions["trace"](
            "http11.send_request_body.complete", {"headers": {"Authorization": "Bearer private"}}
        )
        return httpx.Response(200, stream=DelayedBody())

    client = ServiceClients(Settings(internal_api_key="private"), httpx.MockTransport(handle))
    try:
        assert client.diarize(path, {}) == []
    finally:
        client.close()
    text = caplog.text
    assert text.index("请求体发送完成") < text.index("收到响应头，HTTP 200")
    assert text.index("收到响应头，HTTP 200") < text.index("响应正文已收完")
    assert text.index("响应正文已收完") < text.index("JSON 解析完成")
    assert "结果读取完成，共 0 个说话人时间段" in text
    assert "private" not in text


def test_asr_logs_first_event_before_the_stream_finishes(caplog):
    caplog.set_level("INFO", logger="clients")

    def chunks():
        yield stream_event("你好")
        assert "收到首条 SSE 数据" in caplog.text
        assert "流式正文读取完成" not in caplog.text
        yield stream_event(finish="stop")

    client = streaming_client(ObservedStream(chunks()))
    try:
        assert client.recognize(np.zeros(16000), 16000) == "你好"
    finally:
        client.close()
    assert "流式正文读取完成" in caplog.text


def streaming_client(stream, **settings):
    return ServiceClients(
        Settings(asr_api_url="http://asr/upload", asr_model="local-model", **settings),
        httpx.MockTransport(
            lambda _: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=stream
            )
        ),
    )


def test_runaway_generation_is_closed_before_remaining_events_are_read():
    stream = ObservedStream(
        [stream_event("字" * 300), stream_event("不应读取"), stream_event(finish="stop")]
    )
    client = streaming_client(stream)
    try:
        with pytest.raises(ChunkInferenceError, match="主动关闭生成流"):
            client.recognize(np.zeros(16000), 16000)
    finally:
        client.close()
    assert stream.closed and stream.consumed == 1


def test_token_limit_is_not_accepted_as_a_complete_transcript():
    stream = ObservedStream([stream_event("只有一半"), stream_event(finish="length")])
    client = streaming_client(stream)
    try:
        with pytest.raises(ChunkInferenceError, match="正文被截断"):
            client.recognize(np.zeros(16000), 16000)
    finally:
        client.close()
    assert stream.closed


@pytest.mark.parametrize("done", [b"", b"data: [DONE]\n\n"])
def test_partial_stream_without_normal_finish_is_rejected(done):
    stream = ObservedStream([stream_event("不完整"), done])
    client = streaming_client(stream)
    try:
        with pytest.raises(UpstreamError, match="未收到正常结束标记"):
            client.recognize(np.zeros(16000), 16000)
    finally:
        client.close()


def test_stream_parser_handles_unicode_network_boundaries_comments_and_multiline_data():
    event = json.dumps(
        {"choices": [{"delta": {"content": "你好"}, "finish_reason": None}]}, ensure_ascii=False
    )
    event = event.replace("[{", "[\n{", 1)
    multiline = "\n".join("data: " + line for line in event.splitlines()) + "\n\n"
    wire = b": heartbeat\n\n" + multiline.encode() + stream_event(finish="stop")
    wire += b'data: {"choices": [], "usage": {"completion_tokens": 2}}\n\n'
    wire += b"data: [DONE]\n\n"
    stream = ObservedStream([wire[i : i + 1] for i in range(len(wire))])
    client = streaming_client(stream)
    try:
        assert client.recognize(np.zeros(16000), 16000) == "你好"
    finally:
        client.close()
    assert stream.closed


def test_active_generation_and_heartbeat_obey_elapsed_time_budget(monkeypatch):
    import clients as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])

    def chunks():
        yield stream_event("前半句")
        clock[0] = 91
        yield b": keepalive\n\n"
        yield stream_event(finish="stop")

    stream = ObservedStream(chunks())
    client = streaming_client(stream)
    try:
        with pytest.raises(UpstreamError, match="生成耗时超过 90 秒"):
            client.recognize(np.zeros(16000), 16000)
    finally:
        client.close()
    assert stream.closed and stream.consumed == 2


def test_stream_error_does_not_expose_api_key():
    payload = b'data: {"error": {"message": "failed secret"}}\n\n'
    stream = ObservedStream([payload])
    client = streaming_client(stream, asr_api_key="secret")
    try:
        with pytest.raises(UpstreamError, match="流式错误") as caught:
            client.recognize(np.zeros(16000), 16000)
        assert "secret" not in str(caught.value)
    finally:
        client.close()


@pytest.mark.parametrize(
    "wire", [b"data: invalid-json\n\n", b"data: []\n\n", b'data: {"choices": []}\n\n']
)
def test_invalid_stream_is_not_silently_treated_as_empty_speech(wire):
    client = streaming_client(ObservedStream([wire]))
    try:
        with pytest.raises(UpstreamError):
            client.recognize(np.zeros(16000), 16000)
    finally:
        client.close()


def test_openai_transcript_delta_and_done_events():
    wire = b'data: {"type": "transcript.text.delta", "delta": "Hello"}\n\n'
    wire += b'data: {"type": "transcript.text.done", "text": "Hello!"}\n\n'
    client = streaming_client(ObservedStream([wire]))
    try:
        assert client.recognize(np.zeros(16000), 16000) == "Hello!"
    finally:
        client.close()


def test_asr_generation_limit_and_timeout_can_be_overridden():
    def handle(request):
        assert multipart(request)["max_completion_tokens"][1] == b"512"
        assert request.extensions["timeout"]["read"] == 30
        return asr_response("你好")

    client = ServiceClients(
        Settings(
            asr_api_url="http://asr/upload",
            asr_model="local-model",
            asr_max_completion_tokens=512,
            asr_timeout_seconds=60,
            api_timeout_seconds=30,
        ),
        httpx.MockTransport(handle),
    )
    try:
        assert client.recognize(np.zeros(16000), 16000) == "你好"
    finally:
        client.close()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_asr_token_limits_are_rejected(value):
    with pytest.raises(ValueError, match="正整数"):
        Settings(asr_max_completion_tokens=value)
