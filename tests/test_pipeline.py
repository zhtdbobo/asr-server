import numpy as np
import pytest
import soundfile as sf

from clients import ChunkInferenceError, UpstreamError
from config import Settings
from pipeline import MeetingPipeline, PipelineBusyError, audio_chunks


class FakeServices:
    def __init__(self):
        self.chunks = []

    def diarize(self, path, options):
        assert path.name == "mono-16k.wav"
        info = sf.info(path)
        assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
        self.upload_path = path
        assert options == {"num_speakers": 2}
        return [{"start": 0, "end": 1, "speaker": "B"}, {"start": 31, "end": 33, "speaker": "A"}]

    def recognize(self, waveform, sr):
        self.chunks.append(waveform.copy())
        return "你" if len(self.chunks) == 1 else "好。"

    def align(self, waveform, sr, text, language):
        assert sr == 16000 and language == "Chinese"
        start = 0 if text == "你" else 2
        return [{"text": text.strip("。"), "start": start, "end": start + 0.5}]


def test_external_services_and_global_timestamps(monkeypatch, caplog):
    import pipeline as module

    caplog.set_level("INFO", logger="pipeline")

    waveform = np.ones(16000 * 35, dtype=np.float32)
    monkeypatch.setattr(module, "decode_audio", lambda *_: (waveform, 16000))
    services = FakeServices()
    engine = MeetingPipeline(Settings(asr_api_url="http://asr/v1/audio/transcriptions"), services)
    result = engine.transcribe("fake.wav", num_speakers=2)
    assert result["text"] == "你好。"
    assert result["speaker_count"] == 2
    assert result["segments"][1]["start"] > 29
    assert result["segments"][1]["speaker_name"] == "发言人2"
    assert len(services.chunks) == 2
    assert not services.upload_path.exists()
    messages = "\n".join(caplog.messages)
    assert "[fake.wav] 开始解码音频" in messages
    assert "录音时长 35.0 秒" in messages
    assert "分块 1" in messages and "分块 2" in messages
    assert "开始时间戳对齐" in messages and "转写完成" in messages


def test_chunks_have_no_missing_or_duplicate_samples():
    waveform = np.arange(16000 * 125 + 7, dtype=np.float32)
    chunks = list(audio_chunks(waveform, 16000, 30))
    assert np.array_equal(np.concatenate([chunk for chunk, _ in chunks]), waveform)
    cursor = 0
    for chunk, offset in chunks:
        assert offset == cursor / 16000
        assert len(chunk) <= 30 * 16000
        cursor += len(chunk)


def test_capacity_released_after_failure():
    engine = MeetingPipeline(Settings(job_workers=1), FakeServices())
    engine._slots.acquire()
    with pytest.raises(PipelineBusyError):
        engine.transcribe("unused.wav")
    engine._slots.release()
    with pytest.raises(UpstreamError, match="ASR_API_URL"):
        engine.transcribe("unused.wav")
    assert engine._slots.acquire(blocking=False)
    engine._slots.release()


class RecoveringServices:
    def __init__(self, problem="timestamps"):
        self.problem = problem
        self.recognized_lengths = []
        self.aligned_texts = []

    def diarize(self, path, options):
        self.upload_path = path
        self.upload_info = sf.info(path)
        return [{"start": 0, "end": 20, "speaker": "A"}]

    def recognize(self, waveform, sr):
        self.recognized_lengths.append(len(waveform))
        if self.problem == "network":
            raise UpstreamError("ASR HTTP 503")
        if len(waveform) > 10 * sr:
            if self.problem == "long_text":
                return "异常重复" * 40000
            if self.problem == "truncated":
                raise ChunkInferenceError("ASR 正文被截断")
            return "错误初稿"
        return "你好" if len(self.recognized_lengths) == 2 else "世界"

    def align(self, waveform, sr, text, language):
        self.aligned_texts.append(text)
        if self.problem == "always_bad" or text == "错误初稿":
            return [{"text": text, "start": 0, "end": 999}]
        return [{"text": text, "start": 0, "end": 0.5}]


@pytest.mark.parametrize("problem", ["timestamps", "long_text", "truncated"])
def test_bad_chunk_is_re_recognized_without_partial_text_or_duplicate_samples(monkeypatch, problem):
    import pipeline as module

    waveform = np.zeros(20 * 16000, dtype=np.float32)
    monkeypatch.setattr(module, "decode_audio", lambda *_: (waveform, 16000))
    services = RecoveringServices(problem)
    engine = MeetingPipeline(Settings(asr_api_url="http://asr/upload"), services)
    result = engine.transcribe("original-8ch.wav")
    assert result["text"] == "你好世界"
    assert [(word["start"], word["end"]) for word in result["words"]] == [(0, 0.5), (10, 10.5)]
    assert services.recognized_lengths == [320000, 160000, 160000]
    assert result["warnings"] and "二等分" in result["warnings"][0]
    assert services.upload_info.duration == 20
    assert services.upload_info.channels == 1 and services.upload_info.subtype == "PCM_16"
    assert not services.upload_path.exists()
    if problem == "long_text":
        assert services.aligned_texts == ["你好", "世界"]
        assert "未发送到对齐服务" in result["warnings"][0]


def test_invalid_timestamps_fail_with_values_after_bounded_retries(monkeypatch):
    import pipeline as module

    waveform = np.zeros(20 * 16000, dtype=np.float32)
    monkeypatch.setattr(module, "decode_audio", lambda *_: (waveform, 16000))
    services = RecoveringServices("always_bad")
    engine = MeetingPipeline(Settings(asr_api_url="http://asr/upload"), services)
    with pytest.raises(UpstreamError, match="end=999") as caught:
        engine.transcribe("bad.wav")
    assert "时长=5.000 秒" in str(caught.value)
    assert "原录音 0.000–5.000 秒" in str(caught.value)
    assert services.recognized_lengths == [320000, 160000, 80000]
    assert not services.upload_path.exists()
    assert engine._slots.acquire(blocking=False)
    engine._slots.release()


def test_transport_failures_do_not_trigger_more_audio_subdivision(monkeypatch):
    import pipeline as module

    monkeypatch.setattr(module, "decode_audio", lambda *_: (np.zeros(320000), 16000))
    services = RecoveringServices("network")
    engine = MeetingPipeline(Settings(asr_api_url="http://asr/upload"), services)
    with pytest.raises(UpstreamError, match="HTTP 503"):
        engine.transcribe("bad.wav")
    assert services.recognized_lengths == [320000]


def test_alignment_text_mismatch_is_detected_before_committing_chunk(monkeypatch):
    import pipeline as module

    monkeypatch.setattr(module, "decode_audio", lambda *_: (np.zeros(320000), 16000))

    class MismatchingServices(RecoveringServices):
        def align(self, waveform, sr, text, language):
            if text == "错误初稿":
                return [{"text": "缺字", "start": 0, "end": 0.5}]
            return super().align(waveform, sr, text, language)

    result = MeetingPipeline(
        Settings(asr_api_url="http://asr/upload"), MismatchingServices()
    ).transcribe("mismatch.wav")
    assert result["text"] == "你好世界"
    assert "不一致" in result["warnings"][0]


def test_large_multichannel_input_is_not_uploaded_to_diarization(tmp_path, monkeypatch):
    import pipeline as module

    source = tmp_path / "array.wav"
    sf.write(source, np.zeros((320000, 8), dtype=np.float32), 16000, subtype="PCM_16")
    monkeypatch.setattr(module, "decode_audio", lambda *_: (np.zeros(320000), 16000))

    class SizeCheckingServices(RecoveringServices):
        def diarize(self, path, options):
            assert path != source
            assert path.stat().st_size < source.stat().st_size / 7
            return super().diarize(path, options)

    result = MeetingPipeline(
        Settings(asr_api_url="http://asr/upload"), SizeCheckingServices()
    ).transcribe(source)
    assert result["text"] == "你好世界"
