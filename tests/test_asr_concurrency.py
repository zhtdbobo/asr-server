from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock

import numpy as np
import pytest

from batch_transcribe import run_batch
from clients import ChunkInferenceError, UpstreamError
from config import Settings
from pipeline import MeetingPipeline, PipelineBusyError


def test_parallel_chunks_finish_out_of_order_but_keep_text_speakers_and_samples(monkeypatch):
    waveform = np.arange(67 * 16000, dtype=np.float32)
    monkeypatch.setattr("pipeline.decode_audio", lambda *_: (waveform, 16000))
    second_finished = Event()
    chunks, completed = [], []
    lock = Lock()

    class Services:
        def diarize(self, *_):
            return [
                {"start": 0, "end": 65, "speaker": "A"},
                {"start": 65, "end": 66, "speaker": "B"},
                {"start": 66, "end": 67, "speaker": "A"},
            ]

        def recognize(self, chunk, sample_rate):
            offset = int(chunk[0])
            if offset == 0:
                assert second_finished.wait(5), "第二个分块未并发执行"
            with lock:
                chunks.append(chunk.copy())
                completed.append(offset)
            if offset != 0:
                second_finished.set()
            return f"part{offset}"

    result = MeetingPipeline(
        Settings(asr_api_url="http://asr/upload", asr_workers=2), Services()
    ).transcribe("meeting.wav", language="English")
    assert completed[0] != 0
    chunks.sort(key=lambda chunk: chunk[0])
    assert np.array_equal(np.concatenate(chunks), waveform)
    assert all(len(chunk) <= 30 * 16000 for chunk in chunks)
    assert [(s["speaker"], s["start"], s["end"]) for s in result["segments"]] == [
        ("A", 0, 65),
        ("B", 65, 66),
        ("A", 66, 67),
    ]
    assert result["segments"][0]["text"] == " ".join(
        f"part{int(chunk[0])}" for chunk in chunks if chunk[0] < 65 * 16000
    )
    assert [s["text"] for s in result["segments"][1:]] == ["part1040000", "part1056000"]


@pytest.mark.parametrize("file_workers,asr_workers", [(1, 2), (2, 1), (2, 3)])
def test_file_and_asr_workers_bound_combined_requests(
    tmp_path, monkeypatch, file_workers, asr_workers
):
    source = tmp_path / "audio"
    source.mkdir()
    for index in range(file_workers):
        (source / f"{index}.wav").write_bytes(b"audio")

    def decode(path, *_):
        marker = (int(path.stem) + 1) * 100
        return (
            np.repeat(np.arange(marker, marker + asr_workers * 2, dtype=np.float32), 16000),
            16000,
        )

    monkeypatch.setattr("pipeline.decode_audio", decode)
    barrier = Barrier(file_workers * asr_workers)
    lock = Lock()
    active, peaks = {}, {}
    total_active = total_peak = 0

    class Services:
        def diarize(self, *_):
            return [
                {"start": i, "end": i + 1, "speaker": str(i % 2)} for i in range(asr_workers * 2)
            ]

        def recognize(self, chunk, sample_rate):
            nonlocal total_active, total_peak
            file_id = int(chunk[0]) // 100
            with lock:
                active[file_id] = active.get(file_id, 0) + 1
                peaks[file_id] = max(peaks.get(file_id, 0), active[file_id])
                total_active += 1
                total_peak = max(total_peak, total_active)
            try:
                barrier.wait(timeout=5)
                return f"part{int(chunk[0])}"
            finally:
                with lock:
                    active[file_id] -= 1
                    total_active -= 1

    engine = MeetingPipeline(
        Settings(
            asr_api_url="http://asr/upload", job_workers=file_workers, asr_workers=asr_workers
        ),
        Services(),
    )
    report = run_batch(source, tmp_path / "out", engine, workers=file_workers)
    assert report["completed"] == file_workers and report["failed"] == 0
    assert total_peak == file_workers * asr_workers
    assert set(peaks.values()) == {asr_workers}
    assert total_active == 0


def test_parallel_truncation_retries_stay_within_workers_and_keep_order(monkeypatch):
    waveform = np.arange(40 * 16000, dtype=np.float32)
    monkeypatch.setattr("pipeline.decode_audio", lambda *_: (waveform, 16000))
    barrier, lock = Barrier(2), Lock()
    active = peak = calls = 0

    class Services:
        def diarize(self, *_):
            return [
                {"start": 0, "end": 20, "speaker": "A"},
                {"start": 20, "end": 40, "speaker": "B"},
            ]

        def recognize(self, chunk, sample_rate):
            nonlocal active, peak, calls
            with lock:
                active += 1
                peak = max(peak, active)
                calls += 1
            try:
                if len(chunk) > 10 * sample_rate:
                    barrier.wait(timeout=5)
                    raise ChunkInferenceError("ASR 正文被截断")
                return f"part{int(chunk[0])}"
            finally:
                with lock:
                    active -= 1

    result = MeetingPipeline(
        Settings(asr_api_url="http://asr/upload", asr_workers=2), Services()
    ).transcribe("meeting.wav", language="English")
    assert peak == 2 and active == 0 and calls == 6
    assert [s["text"] for s in result["segments"]] == ["part0 part160000", "part320000 part480000"]
    assert len(result["warnings"]) == 2
    assert "0.000–20.000" in result["warnings"][0]
    assert "20.000–40.000" in result["warnings"][1]


def test_failure_drains_active_asr_before_releasing_file_capacity(monkeypatch):
    monkeypatch.setattr(
        "pipeline.decode_audio", lambda *_: (np.arange(4 * 16000, dtype=np.float32), 16000)
    )
    second_started, failed, release = Event(), Event(), Event()
    calls, finished = [], []
    lock = Lock()

    class Services:
        def diarize(self, *_):
            return [{"start": i, "end": i + 1, "speaker": str(i)} for i in range(4)]

        def recognize(self, chunk, sample_rate):
            offset = int(chunk[0])
            with lock:
                calls.append(offset)
            if offset == 0:
                assert second_started.wait(5)
                failed.set()
                raise UpstreamError("ASR HTTP 503")
            second_started.set()
            assert release.wait(5)
            finished.append(offset)
            return "正文"

    engine = MeetingPipeline(
        Settings(asr_api_url="http://asr/upload", job_workers=1, asr_workers=2), Services()
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(engine.transcribe, "meeting.wav")
        try:
            assert failed.wait(5)
            assert not future.done()
            with pytest.raises(PipelineBusyError):
                engine.transcribe("another.wav")
        finally:
            release.set()
        with pytest.raises(UpstreamError, match="HTTP 503"):
            future.result(timeout=5)
    assert {0, 16000}.issubset(calls)
    assert len(finished) == len(calls) - 1
    assert engine._slots.acquire(blocking=False)
    engine._slots.release()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_asr_workers_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="asr_workers"):
        Settings(asr_workers=value)


def test_asr_workers_can_be_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("ASR_WORKERS", "3")
    assert Settings.from_env().asr_workers == 3
