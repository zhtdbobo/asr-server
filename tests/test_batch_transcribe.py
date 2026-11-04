import json
from pathlib import Path
from threading import Barrier, Lock

import pytest

import batch_transcribe as batch
from config import Settings


class FakePipeline:
    def __init__(self, barrier=None):
        self.calls = []
        self.closed = False
        self.barrier = barrier
        self.lock = Lock()
        self.active = self.peak = 0

    def transcribe(self, path, **options):
        with self.lock:
            self.calls.append((path, options))
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.barrier:
                self.barrier.wait(timeout=5)
            if path.name == "bad.wav":
                raise RuntimeError("ASR unavailable")
            return {"formatted_text": f"发言人1\n00:00\n{path.name}", "segments": []}
        finally:
            with self.lock:
                self.active -= 1

    def close(self):
        self.closed = True


def audio(folder, name):
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake audio")
    return path


def test_folder_outputs_continue_after_failure_and_keep_unique_names(tmp_path):
    source, output = tmp_path / "audio", tmp_path / "audio" / "results"
    for name in ("会议.wav", "会议.MP3", "bad.wav", "sub/会议.wav", "notes.txt"):
        audio(source, name)
    audio(output, "exclude.wav")
    pipeline = FakePipeline()
    report = batch.run_batch(
        source,
        output,
        pipeline,
        recursive=True,
        workers=2,
        options={"language": "Chinese", "num_speakers": 3},
    )
    assert report["total"] == 4 and report["completed"] == 3 and report["failed"] == 1
    assert report == json.loads((output / "batch-summary.json").read_text(encoding="utf-8"))
    for name in ("会议.wav", "会议.MP3", "sub/会议.wav"):
        assert (output / f"{name}.txt").read_text(encoding="utf-8").startswith("发言人1\n00:00")
        result = json.loads((output / f"{name}.json").read_text(encoding="utf-8"))
        assert result["formatted_text"].endswith(Path(name).name)
    failed = next(item for item in report["files"] if item["status"] == "failed")
    assert failed["file"] == "bad.wav" and failed["error"] == "ASR unavailable"
    assert not (output / "bad.wav.json").exists()
    assert all(
        options == {"language": "Chinese", "num_speakers": 3} for _, options in pipeline.calls
    )
    assert not list(output.rglob("*.tmp"))


def test_resume_reprocesses_incomplete_outputs_and_default_scans_top_level(tmp_path):
    source, output = tmp_path / "audio", tmp_path / "result"
    for name in ("one.wav", "two.mp3", "sub/ignored.wav"):
        audio(source, name)
    pipeline = FakePipeline()
    assert batch.run_batch(source, output, pipeline)["completed"] == 2
    (output / "two.mp3.json").unlink()
    pipeline.calls.clear()
    report = batch.run_batch(source, output, pipeline, skip_existing=True)
    assert report["completed"] == report["skipped"] == 1
    assert [path.name for path, _ in pipeline.calls] == ["two.mp3"]


def test_parallelism_is_bounded(tmp_path):
    for number in range(4):
        audio(tmp_path / "audio", f"{number}.wav")
    pipeline = FakePipeline(barrier=Barrier(2))
    report = batch.run_batch(tmp_path / "audio", tmp_path / "out", pipeline, workers=2)
    assert report["completed"] == 4 and report["failed"] == 0
    assert pipeline.peak == 2


def test_cli_failure_exit_and_closes_client(tmp_path, monkeypatch):
    audio(tmp_path / "audio", "bad.wav")
    pipeline = FakePipeline()
    monkeypatch.setattr(
        batch.Settings,
        "from_env",
        lambda: Settings(asr_api_url="http://asr/v1/audio/transcriptions"),
    )

    def create_pipeline(settings):
        assert settings.job_workers == 2
        return pipeline

    monkeypatch.setattr(batch, "MeetingPipeline", create_pipeline)
    assert (
        batch.main(
            [str(tmp_path / "audio"), "--output-dir", str(tmp_path / "out"), "--workers", "2"]
        )
        == 1
    )
    assert pipeline.closed


def test_invalid_or_empty_folder_does_not_call_services(tmp_path):
    pipeline = FakePipeline()
    with pytest.raises(ValueError, match="不是目录"):
        batch.run_batch(tmp_path / "missing", tmp_path / "out", pipeline)
    with pytest.raises(ValueError, match="没有支持的音频"):
        batch.run_batch(tmp_path, tmp_path / "out", pipeline)
    assert pipeline.calls == []


def test_output_directory_can_be_a_parent_of_input(tmp_path):
    source = tmp_path / "audio"
    path = audio(source, "meeting.wav")
    assert batch.discover_audio(source, tmp_path, recursive=True) == [path]
