import json
import logging
from pathlib import Path
from threading import Barrier, Lock

import pytest

import batch_transcribe as batch
from clients import UpstreamError
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
            return {
                "formatted_text": f"发言人1\n00:00\n{path.name}",
                "segments": [
                    {
                        "speaker": "S2",
                        "start": 1,
                        "end": 3,
                        "text": "收到",
                        "speaker_name": "发言人2",
                    },
                    {"speaker": "S1", "start": 0, "end": 2, "text": "你好", "timestamp": "00:00"},
                ],
                "warnings": [],
            }
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
        assert result == [
            {"speaker": "S1", "start": 0, "end": 2, "text": "你好"},
            {"speaker": "S2", "start": 1, "end": 3, "text": "收到"},
        ]
        full = json.loads((output / f"{name}.full.json").read_text(encoding="utf-8"))
        assert full["formatted_text"].endswith(Path(name).name)
        assert full["segments"][0]["speaker_name"] == "发言人2"
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
        assert settings.api_timeout_seconds == 90
        return pipeline

    monkeypatch.setattr(batch, "MeetingPipeline", create_pipeline)
    assert (
        batch.main(
            [
                str(tmp_path / "audio"),
                "--output-dir",
                str(tmp_path / "out"),
                "--workers",
                "2",
                "--timeout-seconds",
                "90",
            ]
        )
        == 1
    )
    assert pipeline.closed
    log = (tmp_path / "out" / "batch.log").read_text(encoding="utf-8")
    assert "Python：" in log and "本次配置：" in log
    assert "[asr-file_0]" in log or "[asr-file_1]" in log
    assert "ASR unavailable" in log and "Traceback" in log
    assert "处理结束" in log


def test_custom_log_file_captures_worker_messages_and_restores_logging(tmp_path, monkeypatch):
    audio(tmp_path / "audio", "meeting.wav")
    previous_level = logging.getLogger().level
    handlers = list(logging.getLogger().handlers)
    monkeypatch.setattr(
        batch.Settings, "from_env", lambda: Settings(asr_api_url="http://asr/upload")
    )
    monkeypatch.setattr(batch, "MeetingPipeline", lambda _: FakePipeline())
    log_file = tmp_path / "logs" / "diagnostic.log"
    assert (
        batch.main(
            [
                str(tmp_path / "audio"),
                "--output-dir",
                str(tmp_path / "out"),
                "--log-file",
                str(log_file),
            ]
        )
        == 0
    )
    log = log_file.read_text(encoding="utf-8")
    assert "[开始] meeting.wav" in log and "三份结果文件写入完成" in log
    assert not (tmp_path / "out" / "batch.log").exists()
    assert logging.getLogger().level == previous_level
    assert logging.getLogger().handlers == handlers


def test_failure_logging_preserves_redacted_error_without_private_transport_cause(tmp_path):
    audio(tmp_path / "audio", "meeting.wav")

    class FailedPipeline:
        def transcribe(self, *_args, **_kwargs):
            try:
                raise RuntimeError("private-transport-secret")
            except RuntimeError as exc:
                raise UpstreamError("服务读取失败，已隐藏敏感信息") from exc

    log_path = tmp_path / "batch.log"
    with batch.batch_logging(log_path):
        report = batch.run_batch(tmp_path / "audio", tmp_path / "out", FailedPipeline())
    log = log_path.read_text(encoding="utf-8")
    assert report["failed"] == 1
    assert "服务读取失败" in log and "Traceback" in log
    assert "private-transport-secret" not in log


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


def test_comparison_json_matches_textgrid_schema_and_skips_blank_segments():
    from textgrid_to_json import parse_textgrid

    reference = parse_textgrid(
        '"ooTextFile"\n"TextGrid"\n0 2 <exists> 1\n"IntervalTier" "F_SPK0093" 0 2 1\n0 2 "你好"\n'
    )
    prediction = {
        "segments": [
            {"speaker": "SPK0093", "start": 0, "end": 2, "text": "你好", "timestamp": "00:00"},
            {"speaker": "SPK0093", "start": 2, "end": 3, "text": " "},
        ]
    }
    assert batch.comparison_segments(prediction) == reference
    assert set(batch.comparison_segments(prediction)[0]) == {"speaker", "start", "end", "text"}


def test_resume_migrates_old_full_json_without_inference_and_preserves_txt(tmp_path):
    source, output = tmp_path / "audio", tmp_path / "result"
    path = audio(source, "会议.wav")
    output.mkdir()
    original = FakePipeline().transcribe(path)
    text_path = output / "会议.wav.txt"
    text_path.write_bytes(b"original TXT\r\n")
    (output / "会议.wav.json").write_text(json.dumps(original), encoding="utf-8")
    pipeline = FakePipeline()
    report = batch.run_batch(source, output, pipeline, skip_existing=True)
    assert pipeline.calls == [] and report["skipped"] == 1
    assert report["files"][0]["json_updated"] is True
    assert text_path.read_bytes() == b"original TXT\r\n"
    assert json.loads((output / "会议.wav.json").read_text(encoding="utf-8")) == (
        batch.comparison_segments(original)
    )
    assert json.loads((output / "会议.wav.full.json").read_text(encoding="utf-8")) == original
    assert batch.run_batch(source, output, pipeline, skip_existing=True)["skipped"] == 1
    assert pipeline.calls == []


def test_resume_reprocesses_invalid_cached_json(tmp_path):
    source, output = tmp_path / "audio", tmp_path / "result"
    path = audio(source, "meeting.wav")
    output.mkdir()
    (output / "meeting.wav.txt").write_text("incomplete", encoding="utf-8")
    (output / "meeting.wav.json").write_text('{"segments": null}', encoding="utf-8")
    pipeline = FakePipeline()
    assert batch.run_batch(source, output, pipeline, skip_existing=True)["completed"] == 1
    assert pipeline.calls == [(path, {})]


def test_waiting_progress_is_visible_before_any_file_finishes(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO", logger="batch_transcribe")
    audio(tmp_path / "audio", "meeting.wav")
    actual_wait = batch.wait
    calls = []

    def first_wait_times_out(futures, **kwargs):
        calls.append(kwargs["timeout"])
        if len(calls) == 1:
            return set(), set(futures)
        return actual_wait(futures, **kwargs)

    monkeypatch.setattr(batch, "wait", first_wait_times_out)
    assert batch.run_batch(tmp_path / "audio", tmp_path / "out", FakePipeline())["completed"] == 1
    assert "[等待]" in caplog.text
    assert calls[0] == 15
