import json
import shutil
from email import policy
from email.parser import BytesParser

import httpx
import pytest

import batch_transcribe_moss as moss
from batch_transcribe import comparison_segments, run_batch
from clients import UpstreamError
from compare_transcripts import compare_segments, load_segments


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


def response_body():
    return {
        "task": "transcribe",
        "duration": 9.5,
        "text": "你好，开始讨论。好的！收到。",
        "segments": [
            {
                "type": "transcript.text.segment",
                "id": "seg_2",
                "speaker": "S01",
                "start": 7.25,
                "end": 9.5,
                "text": "收到。",
            },
            {
                "type": "transcript.text.segment",
                "id": "seg_0",
                "speaker": "S01",
                "start": 0.125,
                "end": 3.2,
                "text": "你好，开始讨论。",
            },
            {
                "type": "transcript.text.segment",
                "id": "seg_1",
                "speaker": "S02",
                "start": 2.8,
                "end": 5.5,
                "text": "好的！",
            },
        ],
        "usage": {"type": "duration", "seconds": 10},
    }


def audio(folder, name="会议.wav"):
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"original audio bytes")
    return path


@pytest.fixture
def fake_inspection(monkeypatch):
    def inspected(path, *_limits):
        return {"duration_seconds": 9.5, "size_bytes": path.stat().st_size}

    monkeypatch.setattr(moss, "inspect_audio", inspected)


def install_mock_pipeline(monkeypatch, handler):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    for name in ("MOSS_API_URL", "MOSS_MODEL", "MOSS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    actual = moss.MossPipeline
    instances = []

    def factory(*args, **kwargs):
        client = actual(*args, **kwargs, transport=httpx.MockTransport(handler))
        instances.append(client)
        return client

    monkeypatch.setattr(moss, "MossPipeline", factory)
    return instances


@pytest.mark.usefixtures("fake_inspection")
def test_cli_upload_outputs_and_reference_comparison(tmp_path, monkeypatch):
    source = tmp_path / "audio"
    path = audio(source, "sub/会议.wav")
    body = response_body()
    calls = []

    def handle(request):
        calls.append(request)
        assert str(request.url) == "http://moss:8000/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer test-secret"
        parts = multipart(request)
        assert parts["file"] == (path.name, path.read_bytes())
        assert parts["model"][1] == b"local-moss"
        assert parts["response_format"][1] == b"diarized_json"
        assert parts["temperature"][1] == b"0"
        assert parts["max_completion_tokens"][1] == b"16384"
        assert parts["language"][1] == b"zh"
        assert parts["prompt"][1].decode() == "热词：项目"
        assert "stream" not in parts
        assert request.extensions["timeout"]["read"] == 1200
        return httpx.Response(200, json=body)

    instances = install_mock_pipeline(monkeypatch, handle)
    monkeypatch.setenv("MOSS_API_KEY", "test-secret")
    monkeypatch.setenv("MOSS_API_URL", "http://ignored:8000/v1/audio/transcriptions")
    root = tmp_path / "results"
    assert (
        moss.main(
            [
                str(source),
                "--output-dir",
                str(root),
                "--recursive",
                "--workers",
                "2",
                "--api-url",
                "http://moss:8000/v1/audio/transcriptions",
                "--model",
                "local-moss",
                "--timeout-seconds",
                "1200",
                "--max-completion-tokens",
                "16384",
                "--language",
                "zh",
                "--prompt",
                "热词：项目",
            ]
        )
        == 0
    )
    (output,) = root.iterdir()
    full = json.loads((output / "sub/会议.wav.full.json").read_text(encoding="utf-8"))
    assert all(full[key] == value for key, value in body.items())
    assert full["speakers"] == [{"id": "S01", "name": "发言人1"}, {"id": "S02", "name": "发言人2"}]
    expected = [
        {"speaker": "S01", "start": 0.125, "end": 3.2, "text": "你好，开始讨论。"},
        {"speaker": "S02", "start": 2.8, "end": 5.5, "text": "好的！"},
        {"speaker": "S01", "start": 7.25, "end": 9.5, "text": "收到。"},
    ]
    simple = output / "sub/会议.wav.json"
    assert load_segments(simple) == expected
    assert load_segments(output / "sub/会议.wav.full.json") == expected
    text = (output / "sub/会议.wav.txt").read_text(encoding="utf-8")
    assert (
        text
        == "发言人1\n00:00\n你好，开始讨论。\n\n发言人2\n00:02\n好的！\n\n发言人1\n00:07\n收到。\n"
    )
    # Verify that downstream comparison accepts anonymous labels and overlap unchanged.
    reference = [
        {**s, "speaker": {"S01": "SPK0093", "S02": "SPK0094"}[s["speaker"]]} for s in expected
    ]
    compared = compare_segments(reference, load_segments(simple))
    assert compared["text_score"]["cer"] == compared["diarization"]["der"] == 0
    report = json.loads((output / "batch-summary.json").read_text(encoding="utf-8"))
    assert report["completed"] == 1 and report["failed"] == 0
    assert report["files"][0]["duration_seconds"] == 9.5
    assert set(report["timing_summary"]["stages"]) == set(moss.MOSS_TIMING_LABELS)
    assert report["timing_summary"]["stages"]["request_seconds"]["count"] == 1
    timing = (output / "timing-report.txt").read_text(encoding="utf-8")
    assert "MOSS 上传、等待及联合推理" in timing and "说话人分离：" not in timing
    assert "test-secret" not in (output / "batch.log").read_text(encoding="utf-8")
    assert len(calls) == 1 and instances[0].http.is_closed
    # Resuming reads the existing comparison files without repeating inference.
    assert moss.main([str(source), "--resume-dir", str(output), "--recursive"]) == 0
    assert len(calls) == 1 and len(list(root.iterdir())) == 1
    resumed = json.loads((output / "batch-summary.json").read_text(encoding="utf-8"))
    assert resumed["skipped"] == 1
    assert resumed["timing_summary"]["stages"]["request_seconds"]["count"] == 0
    assert all(instance.http.is_closed for instance in instances)


@pytest.mark.usefixtures("fake_inspection")
def test_failure_continues_batch_redacts_key_and_records_request_time(tmp_path, monkeypatch):
    source = tmp_path / "audio"
    audio(source, "bad.wav")
    audio(source, "good.mp3")
    calls = []

    def handle(request):
        filename = multipart(request)["file"][0]
        calls.append(filename)
        if filename == "bad.wav":
            return httpx.Response(400, json={"error": {"message": "failed test-secret"}})
        return httpx.Response(200, json=response_body())

    instances = install_mock_pipeline(monkeypatch, handle)
    monkeypatch.setenv("MOSS_API_KEY", "test-secret")
    assert moss.main([str(source), "--output-dir", str(tmp_path / "out")]) == 1
    (output,) = (tmp_path / "out").iterdir()
    report_text = (output / "batch-summary.json").read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["completed"] == report["failed"] == 1
    assert sorted(calls) == ["bad.wav", "good.mp3"]
    assert (output / "good.mp3.json").is_file()
    assert not (output / "bad.wav.json").exists()
    assert report["timing_summary"]["stages"]["request_seconds"]["count"] == 2
    log = (output / "batch.log").read_text(encoding="utf-8")
    assert "test-secret" not in report_text + log
    assert "[redacted]" in log and "HTTP 400" in log
    assert all(instance.http.is_closed for instance in instances)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"text": "[0.1][S01]你好[1.2]"}),
        httpx.Response(200, json={"text": "你好", "segments": []}),
        httpx.Response(
            200, json={"segments": [{"speaker": None, "start": 0, "end": 1, "text": "你好"}]}
        ),
        httpx.Response(
            200, json={"segments": [{"speaker": "S01", "start": 2, "end": 1, "text": "你好"}]}
        ),
        httpx.Response(200, json={"segments": [], "finish_reason": "length"}),
        httpx.Response(200, text="invalid JSON"),
        httpx.Response(200, json=[]),
        httpx.Response(503, text="service unavailable"),
    ],
)
@pytest.mark.usefixtures("fake_inspection")
def test_bad_responses_do_not_create_comparison_files(tmp_path, response):
    source, output = tmp_path / "audio", tmp_path / "out"
    audio(source)
    calls = []

    def handle(request):
        calls.append(request)
        return response

    client = moss.MossPipeline(
        "http://moss/v1/audio/transcriptions", transport=httpx.MockTransport(handle)
    )
    try:
        report = run_batch(source, output, client, timing_labels=moss.MOSS_TIMING_LABELS)
    finally:
        client.close()
    assert len(calls) == 1 and report["failed"] == 1
    assert not (output / "会议.wav.json").exists()


@pytest.mark.usefixtures("fake_inspection")
def test_timeout_is_not_retried_and_private_cause_is_not_logged(tmp_path, caplog):
    caplog.set_level("INFO")
    path = audio(tmp_path / "audio")
    calls = []

    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("private-transport-secret", request=request)

    client = moss.MossPipeline(
        "http://moss/v1/audio/transcriptions", transport=httpx.MockTransport(handle)
    )
    try:
        with pytest.raises(UpstreamError, match="未自动重复提交") as caught:
            client.transcribe(path)
    finally:
        client.close()
    assert len(calls) == 1
    assert caught.value.timings["request_seconds"] >= 0
    assert "private-transport-secret" not in caplog.text + str(caught.value)


def test_silence_and_optional_duration_do_not_invent_speakers():
    body = {"text": "", "segments": []}
    result = moss.prepare_result(body)
    assert result["formatted_text"] == "" and result["speakers"] == []
    assert comparison_segments(result) == []


@pytest.mark.parametrize(
    "args",
    [
        ["--workers", "0"],
        ["--max-completion-tokens", "0"],
        ["--timeout-seconds", "nan"],
        ["--api-url", "http://moss:8000"],
        ["--max-upload-mb", "nan"],
        ["--max-audio-seconds", "-1"],
    ],
)
def test_cli_rejects_invalid_options_without_output(tmp_path, monkeypatch, args):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    with pytest.raises(SystemExit) as caught:
        moss.main([str(tmp_path), "--output-dir", str(tmp_path / "out"), *args])
    assert caught.value.code in (1, 2)
    assert not (tmp_path / "out").exists()


def test_original_audio_is_uploaded_byte_for_byte_without_resampling(tmp_path):
    import numpy as np
    import soundfile as sf

    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is required for the audio integration check")
    path = tmp_path / "stereo.wav"
    waveform = np.sin(2 * np.pi * 440 * np.arange(96000) / 48000) * 0.2
    sf.write(path, np.column_stack((waveform, waveform)), 48000, subtype="PCM_24")
    original = path.read_bytes()
    calls = []

    def handle(request):
        calls.append(request)
        assert multipart(request)["file"] == (path.name, original)
        return httpx.Response(200, json={"text": "", "segments": []})

    client = moss.MossPipeline(
        "http://moss/v1/audio/transcriptions", transport=httpx.MockTransport(handle)
    )
    try:
        result = client.transcribe(path)
    finally:
        client.close()
    assert len(calls) == 1 and path.read_bytes() == original
    assert result["duration"] == pytest.approx(2)
    assert result["audio_input"]["sample_rate"] == "48000"
    assert result["audio_input"]["channels"] == 2
    assert sf.info(path).subtype == "PCM_24"


@pytest.mark.parametrize(
    "max_mb,max_seconds,match",
    [
        (0.00001, 2, "超过上传上限"),
        (25, 0.5, "未自动切块"),
    ],
)
def test_limit_failures_never_upload_audio(tmp_path, max_mb, max_seconds, match):
    import numpy as np
    import soundfile as sf

    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is required for the audio integration check")
    source = tmp_path / "audio"
    source.mkdir()
    path = source / "meeting.wav"
    sf.write(path, np.zeros(32000), 16000, subtype="PCM_16")
    original = path.read_bytes()
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=response_body())

    client = moss.MossPipeline(
        "http://moss/v1/audio/transcriptions",
        max_upload_mb=max_mb,
        max_audio_seconds=max_seconds,
        transport=httpx.MockTransport(handle),
    )
    try:
        report = run_batch(source, tmp_path / "out", client, timing_labels=moss.MOSS_TIMING_LABELS)
    finally:
        client.close()
    assert report["failed"] == 1 and match in report["files"][0]["error"]
    assert calls == [] and path.read_bytes() == original
    assert report["timing_summary"]["stages"]["request_seconds"]["count"] == 0
