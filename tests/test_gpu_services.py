import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from config import Settings
from gpu_services import GPUEngine, create_gpu_app


class FakeEngine:
    loaded = False

    def __init__(self):
        self.calls = []

    def load(self):
        self.loaded = True

    def run(self, path, **options):
        self.calls.append((Path(path), options, Path(path).read_bytes()))
        return {"turns": [{"start": 1.2, "end": 3.4, "speaker": "A"}]}


def test_diarization_service_auth_timestamps_and_cleanup():
    engine = FakeEngine()
    app = create_gpu_app(engine, Settings(internal_api_key="private"))
    with TestClient(app) as client:
        assert client.get("/health").json()["models_loaded"] is True
        assert (
            client.post("/v1/diarization", files={"file": ("a.wav", b"audio")}).status_code == 401
        )
        response = client.post(
            "/v1/diarization",
            data={"num_speakers": "3"},
            files={"file": ("a.wav", b"audio")},
            headers={"Authorization": "Bearer private"},
        )
        assert response.status_code == 200
        assert response.json()["turns"] == [{"start": 1.2, "end": 3.4, "speaker": "A"}]
        assert engine.calls[0][1] == {"num_speakers": 3}
        assert not engine.calls[0][0].exists()
        assert "/v1/alignment" not in client.get("/openapi.json").json()["paths"]


def test_gpu_parameter_validation():
    engine = FakeEngine()
    with TestClient(create_gpu_app(engine, Settings())) as client:
        response = client.post(
            "/v1/diarization",
            files={"file": ("a.wav", b"audio")},
            data={"min_speakers": 4, "max_speakers": 2},
        )
        assert response.status_code == 422
        assert engine.calls == []


def test_model_loader_uses_selected_gpu_without_importing_qwen(monkeypatch, tmp_path):
    seen = {}
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True), device=lambda value: value
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", None)

    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen.update(path=path, kwargs=kwargs)
            return SimpleNamespace(to=lambda device: seen.update(device=device))

    pyannote = ModuleType("pyannote")
    pyannote.audio = SimpleNamespace(Pipeline=Loader)
    monkeypatch.setitem(sys.modules, "pyannote", pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote.audio)
    diarization_path = tmp_path / "community-1"
    diarization_path.mkdir()
    (diarization_path / "config.yaml").write_text("pipeline: {}", encoding="utf-8")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.setenv(name, "0")
    settings = Settings(diarization_device="cuda:1", diarization_model=str(diarization_path))
    engine = GPUEngine(settings)
    engine.load()
    assert engine.loaded
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        assert __import__("os").environ[name] == "1"
    assert seen["device"] == "cuda:1"
    assert seen["path"] == settings.diarization_model


@pytest.mark.parametrize("directory_exists", [False, True])
def test_missing_local_model_fails_before_importing_inference_libraries(
    monkeypatch, tmp_path, directory_exists
):
    model_path = tmp_path / "missing-model"
    if directory_exists:
        model_path.mkdir()
    monkeypatch.setitem(sys.modules, "torch", None)
    engine = GPUEngine(Settings(diarization_model=str(model_path)))
    with pytest.raises(RuntimeError, match="完整本地模型目录"):
        engine.load()
    assert not engine.loaded


@pytest.mark.parametrize("exclusive_available", [False, True])
def test_inference_uses_exclusive_turns_and_preserves_timestamps(monkeypatch, exclusive_available):
    import gpu_services as module

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            inference_mode=nullcontext,
            from_numpy=lambda _: SimpleNamespace(unsqueeze=lambda _: "tensor"),
        ),
    )
    monkeypatch.setattr(module, "decode_audio", lambda *_: (np.zeros(16000), 16000))
    exclusive = SimpleNamespace(
        itertracks=lambda **_: [(SimpleNamespace(start=0.125, end=0.875), None, "SPEAKER_00")]
    )
    result = (
        SimpleNamespace(exclusive_speaker_diarization=exclusive) if exclusive_available else None
    )
    engine = GPUEngine(Settings())
    engine.model = lambda *_, **__: result
    if exclusive_available:
        assert engine.run("audio.wav") == {
            "turns": [{"start": 0.125, "end": 0.875, "speaker": "SPEAKER_00"}]
        }
    else:
        with pytest.raises(RuntimeError, match="exclusive_speaker_diarization"):
            engine.run("audio.wav")
