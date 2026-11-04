import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from config import Settings
from gpu_services import GPUEngine, create_gpu_app


class FakeEngine:
    loaded = False

    def __init__(self, kind):
        self.kind = kind
        self.calls = []

    def load(self):
        self.loaded = True

    def run(self, path, **options):
        self.calls.append((Path(path), options, Path(path).read_bytes()))
        return {"turns": []} if self.kind == "diarization" else {"words": []}


@pytest.mark.parametrize(
    "kind,route,data,expected",
    [
        ("diarization", "/v1/diarization", {"num_speakers": "3"}, {"num_speakers": 3}),
        (
            "aligner",
            "/v1/alignment",
            {"text": "你好", "language": "Chinese"},
            {"text": "你好", "language": "Chinese"},
        ),
    ],
)
def test_gpu_services_start_separately_and_cleanup(kind, route, data, expected):
    engine = FakeEngine(kind)
    app = create_gpu_app(kind, engine, Settings(internal_api_key="private"))
    with TestClient(app) as client:
        assert client.get("/health").json()["models_loaded"] is True
        assert client.post(route, data=data, files={"file": ("a.wav", b"audio")}).status_code == 401
        response = client.post(
            route,
            data=data,
            files={"file": ("a.wav", b"audio")},
            headers={"Authorization": "Bearer private"},
        )
        assert response.status_code == 200
        assert engine.calls[0][1] == expected
        assert not engine.calls[0][0].exists()


def test_gpu_parameter_validation():
    engine = FakeEngine("diarization")
    with TestClient(create_gpu_app("diarization", engine, Settings())) as client:
        response = client.post(
            "/v1/diarization",
            files={"file": ("a.wav", b"audio")},
            data={"min_speakers": 4, "max_speakers": 2},
        )
        assert response.status_code == 422
        assert engine.calls == []


@pytest.mark.parametrize("kind", ["diarization", "aligner"])
def test_model_loaders_use_the_selected_gpu_without_loading_asr(monkeypatch, tmp_path, kind):
    seen = {}
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: True,
            device=lambda _: nullcontext(),
        ),
        device=lambda value: value,
        float32="float32",
        bfloat16="bfloat16",
        float16="float16",
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen.update(path=path, kwargs=kwargs)
            return SimpleNamespace(to=lambda device: seen.update(device=device))

    pyannote = ModuleType("pyannote")
    pyannote.audio = SimpleNamespace(Pipeline=Loader)
    monkeypatch.setitem(sys.modules, "pyannote", pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote.audio)
    # Expose only ForcedAligner; loading a local Qwen3ASRModel would fail this test.
    monkeypatch.setitem(sys.modules, "qwen_asr", SimpleNamespace(Qwen3ForcedAligner=Loader))
    diarization_path = tmp_path / "community-1"
    diarization_path.mkdir()
    (diarization_path / "config.yaml").write_text("pipeline: {}", encoding="utf-8")
    aligner_path = tmp_path / "aligner"
    aligner_path.mkdir()
    (aligner_path / "config.json").write_text("{}", encoding="utf-8")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.setenv(name, "0")
    settings = Settings(
        diarization_device="cuda:1",
        aligner_device="cuda:2",
        diarization_model=str(diarization_path),
        aligner_model=str(aligner_path),
    )
    engine = GPUEngine(kind, settings)
    engine.load()
    assert engine.loaded
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        assert __import__("os").environ[name] == "1"
    if kind == "diarization":
        assert seen["device"] == "cuda:1"
        assert seen["path"] == settings.diarization_model
    else:
        assert seen["kwargs"]["device_map"] == "cuda:2"
        assert seen["kwargs"]["dtype"] == "bfloat16"
        assert seen["kwargs"]["local_files_only"] is True
        assert seen["path"] == settings.aligner_model


@pytest.mark.parametrize("kind", ["diarization", "aligner"])
@pytest.mark.parametrize("directory_exists", [False, True])
def test_missing_local_models_fail_before_importing_inference_libraries(
    monkeypatch, tmp_path, kind, directory_exists
):
    model_path = tmp_path / "missing-model"
    if directory_exists:
        model_path.mkdir()
    monkeypatch.setitem(sys.modules, "torch", None)
    settings = Settings(**{f"{kind}_model": str(model_path)})
    engine = GPUEngine(kind, settings)
    with pytest.raises(RuntimeError, match="完整本地模型目录"):
        engine.load()
    assert not engine.loaded
