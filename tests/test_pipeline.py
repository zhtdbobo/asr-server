from pathlib import Path

import numpy as np
import pytest

from clients import UpstreamError
from config import Settings
from pipeline import MeetingPipeline, PipelineBusyError, audio_chunks


class FakeServices:
    def __init__(self):
        self.chunks = []

    def diarize(self, path, options):
        assert path == Path("fake.wav")
        assert options == {"num_speakers": 2}
        return [{"start": 0, "end": 1, "speaker": "B"}, {"start": 31, "end": 33, "speaker": "A"}]

    def recognize(self, waveform, sr):
        self.chunks.append(waveform.copy())
        return "你" if len(self.chunks) == 1 else "好。"

    def align(self, waveform, sr, text, language):
        assert sr == 16000 and language == "Chinese"
        start = 0 if text == "你" else 2
        return [{"text": text.strip("。"), "start": start, "end": start + 0.5}]


def test_external_services_and_global_timestamps(monkeypatch):
    import pipeline as module

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
