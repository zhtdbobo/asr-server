import numpy as np
import pytest
import soundfile as sf

from clients import ChunkInferenceError, UpstreamError
from config import Settings
from pipeline import MeetingPipeline, PipelineBusyError, audio_chunks


class FakeServices:
    def __init__(self, turns=None, texts=None):
        self.chunks = []
        self.turns = (
            turns
            if turns is not None
            else [
                {"start": 0, "end": 1, "speaker": "B"},
                {"start": 31, "end": 33, "speaker": "A"},
            ]
        )
        self.texts = texts or ["你", "好。"]

    def diarize(self, path, options):
        assert path.name == "mono-16k.wav"
        info = sf.info(path)
        assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
        self.upload_path = path
        self.speaker_options = options
        return self.turns

    def recognize(self, waveform, sr):
        assert sr == 16000
        self.chunks.append(waveform.copy())
        return self.texts[(len(self.chunks) - 1) % len(self.texts)]


def test_external_services_and_diarization_timestamps_without_aligner(monkeypatch, caplog):
    import pipeline as module

    caplog.set_level("INFO", logger="pipeline")
    waveform = np.arange(16000 * 35, dtype=np.float32)
    monkeypatch.setattr(module, "decode_audio", lambda *_: (waveform, 16000))
    services = FakeServices()
    engine = MeetingPipeline(Settings(asr_api_url="http://asr/v1/audio/transcriptions"), services)
    result = engine.transcribe("fake.wav", num_speakers=2)
    assert result["text"] == "你好。"
    assert result["speaker_count"] == 2
    assert result["segments"][1]["start"] == 31
    assert result["segments"][1]["end"] == 33
    assert result["segments"][1]["speaker_name"] == "发言人2"
    assert result["timestamp_mode"] == "segment" and result["words"] == []
    assert len(services.chunks) == 2
    assert np.array_equal(services.chunks[0], waveform[:16000])
    assert np.array_equal(services.chunks[1], waveform[31 * 16000 : 33 * 16000])
    assert services.speaker_options == {"num_speakers": 2}
    assert not services.upload_path.exists()
    messages = "\n".join(caplog.messages)
    assert "[fake.wav] 开始解码音频" in messages
    assert "录音时长 35.0 秒" in messages
    assert "说话人片段 1/2" in messages and "说话人片段 2/2" in messages
    assert "转写完成" in messages


def test_speaker_changes_within_one_asr_chunk_are_transcribed_separately(monkeypatch):
    import pipeline as module

    waveform = np.arange(16000 * 3, dtype=np.float32)
    monkeypatch.setattr(module, "decode_audio", lambda *_: (waveform, 16000))
    services = FakeServices(
        turns=[
            {"start": 2, "end": 3, "speaker": "A"},
            {"start": 0, "end": 1, "speaker": "A"},
            {"start": 1, "end": 2, "speaker": "B"},
        ],
        texts=["你好。", "嗯。", "继续。"],
    )
    result = MeetingPipeline(Settings(asr_api_url="http://asr/upload"), services).transcribe(
        "a.wav"
    )
    assert [s["speaker"] for s in result["segments"]] == ["A", "B", "A"]
    assert [s["text"] for s in result["segments"]] == services.texts
    for index, chunk in enumerate(services.chunks):
        assert np.array_equal(chunk, waveform[index * 16000 : (index + 1) * 16000])


def test_long_turn_subdivision_keeps_all_samples_and_original_turn_times(monkeypatch):
    import pipeline as module

    waveform = np.arange(16000 * 72, dtype=np.float32)
    monkeypatch.setattr(module, "decode_audio", lambda *_: (waveform, 16000))
    services = FakeServices(
        turns=[{"start": 1.25, "end": 70.75, "speaker": "A"}],
        texts=["Hello,", "world!", "OK."],
    )
    result = MeetingPipeline(Settings(asr_api_url="http://asr/upload"), services).transcribe(
        "a.wav", language="English"
    )
    assert len(services.chunks) == 3
    assert np.array_equal(np.concatenate(services.chunks), waveform[20000:1132000])
    assert all(len(chunk) <= 30 * 16000 for chunk in services.chunks)
    assert result["text"] == "Hello, world! OK."
    assert len(result["segments"]) == 1
    assert result["segments"][0]["start"] == 1.25
    assert result["segments"][0]["end"] == 70.75


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
    def __init__(self, problem="truncated", offset=0):
        self.problem = problem
        self.offset = offset
        self.recognized_lengths = []

    def diarize(self, path, options):
        self.upload_path = path
        self.upload_info = sf.info(path)
        return [{"start": self.offset, "end": self.offset + 20, "speaker": "A"}]

    def recognize(self, waveform, sr):
        self.recognized_lengths.append(len(waveform))
        if self.problem == "network":
            raise UpstreamError("ASR HTTP 503")
        if self.problem == "always_bad":
            raise ChunkInferenceError("ASR 正文被截断")
        if len(waveform) > 10 * sr:
            if self.problem == "long_text":
                return "异常重复" * 40000
            raise ChunkInferenceError("ASR 正文被截断")
        return "你好" if len(self.recognized_lengths) == 2 else "世界"


@pytest.mark.parametrize("problem", ["long_text", "truncated"])
@pytest.mark.parametrize("offset", [0, 7])
def test_bad_chunk_is_re_recognized_without_partial_text(monkeypatch, problem, offset):
    import pipeline as module

    waveform = np.zeros((20 + offset) * 16000, dtype=np.float32)
    monkeypatch.setattr(module, "decode_audio", lambda *_: (waveform, 16000))
    services = RecoveringServices(problem, offset)
    engine = MeetingPipeline(Settings(asr_api_url="http://asr/upload"), services)
    result = engine.transcribe("original-8ch.wav")
    assert result["text"] == "你好世界"
    assert [(s["start"], s["end"]) for s in result["segments"]] == [(offset, offset + 20)]
    assert result["words"] == []
    assert services.recognized_lengths == [320000, 160000, 160000]
    assert result["warnings"] and "二等分" in result["warnings"][0]
    assert f"{offset:.3f}–{offset + 20:.3f}" in result["warnings"][0]
    assert services.upload_info.duration == 20 + offset
    assert services.upload_info.channels == 1 and services.upload_info.subtype == "PCM_16"
    assert not services.upload_path.exists()


def test_repeated_truncation_fails_after_bounded_retries(monkeypatch):
    import pipeline as module

    monkeypatch.setattr(module, "decode_audio", lambda *_: (np.zeros(320000), 16000))
    services = RecoveringServices("always_bad")
    engine = MeetingPipeline(Settings(asr_api_url="http://asr/upload"), services)
    with pytest.raises(UpstreamError, match="缩块重试已达上限") as caught:
        engine.transcribe("bad.wav")
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


@pytest.mark.parametrize(
    "turns",
    [
        [{"start": -1, "end": 1, "speaker": "A"}],
        [{"start": 0, "end": 999, "speaker": "A"}],
        [{"start": 0, "end": 1, "speaker": None}],
        [{"start": 0, "end": 1}],
        [{"start": 0, "end": 1, "speaker": ""}],
        [{"start": 0, "end": 2, "speaker": "A"}, {"start": 1, "end": 3, "speaker": "B"}],
    ],
)
def test_invalid_turns_fail_before_asr_and_release_capacity(monkeypatch, turns):
    import pipeline as module

    monkeypatch.setattr(module, "decode_audio", lambda *_: (np.zeros(320000), 16000))
    services = FakeServices(turns=turns)
    engine = MeetingPipeline(Settings(asr_api_url="http://asr/upload", job_workers=1), services)
    with pytest.raises(UpstreamError, match="无效的时间区间"):
        engine.transcribe("bad.wav")
    assert services.chunks == []
    assert not services.upload_path.exists()
    assert engine._slots.acquire(blocking=False)
    engine._slots.release()


def test_no_speech_does_not_call_asr(monkeypatch):
    import pipeline as module

    monkeypatch.setattr(module, "decode_audio", lambda *_: (np.zeros(16000), 16000))
    services = FakeServices(turns=[])
    result = MeetingPipeline(Settings(asr_api_url="http://asr/upload"), services).transcribe(
        "a.wav"
    )
    assert services.chunks == []
    assert result["segments"] == result["words"] == []
    assert "未检测到有效语音。" in result["warnings"]


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


def test_chunk_limit_no_longer_depends_on_aligner():
    assert Settings(asr_chunk_seconds=240).asr_chunk_seconds == 240


@pytest.mark.parametrize("problem", ["truncated", "network"])
def test_stage_timings_cover_retries_and_failed_stage_waiting(monkeypatch, problem):
    import pipeline as module

    clock = [0.0]
    monkeypatch.setattr(module, "perf_counter", lambda: clock[0])

    def decode(*_):
        clock[0] += 2
        return np.zeros(320000), 16000

    monkeypatch.setattr(module, "decode_audio", decode)
    original_build = module.build_transcript

    def assemble(**options):
        result = original_build(**options)
        clock[0] += 1
        return result

    monkeypatch.setattr(module, "build_transcript", assemble)

    class TimedServices(RecoveringServices):
        def diarize(self, path, options):
            clock[0] += 3
            return super().diarize(path, options)

        def recognize(self, waveform, sr):
            clock[0] += 7 if problem == "network" else [1, 4, 5][len(self.recognized_lengths)]
            return super().recognize(waveform, sr)

    engine = MeetingPipeline(Settings(asr_api_url="http://asr/upload"), TimedServices(problem))
    if problem == "network":
        with pytest.raises(UpstreamError) as caught:
            engine.transcribe("audio.wav")
        assert caught.value.timings == {
            "decode_seconds": 2,
            "diarization_seconds": 3,
            "asr_seconds": 7,
        }
        assert caught.value.processing_seconds == 12
        assert caught.value.duration_seconds == 20
    else:
        result = engine.transcribe("audio.wav")
        assert result["timings"] == {
            "decode_seconds": 2,
            "diarization_seconds": 3,
            "asr_seconds": 10,
            "assembly_seconds": 1,
        }
        assert result["processing_seconds"] == 16
