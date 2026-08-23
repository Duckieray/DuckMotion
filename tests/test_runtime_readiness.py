from __future__ import annotations

from pathlib import Path

from ltx_backend import LTX25IsolatedBackend
from model_runtime import VideoBackendResolver, describe_video_model
from runtime_probe import probe_python_runtime
from wan_backend import WanDiffusersBackend


def test_wan_t2v_is_now_runnable_by_isolated_backend():
    descriptor = describe_video_model("Wan-AI/Wan2.2-T2V-A14B-Diffusers")

    assert descriptor.capabilities.text_to_video is True
    assert descriptor.capabilities.image_to_video is False
    assert descriptor.backend == "wan_diffusers"
    assert descriptor.supported is True


def test_missing_runtime_python_fails_without_model_loading(tmp_path):
    result = probe_python_runtime(
        str(tmp_path / "missing-python"),
        (("diffusers", "WanPipeline"),),
    )
    assert result["ready"] is False
    assert "does not exist" in result["reason"]


def test_wan_readiness_reports_worker_cuda_diagnostics(monkeypatch):
    descriptor = describe_video_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    backend = WanDiffusersBackend()
    captured = []

    def fake_probe(python, symbols, **_kwargs):
        captured.append((python, tuple(symbols)))
        return {
            "ready": True,
            "reason": None,
            "cuda_available": True,
            "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
            "vram_gb": 15.9,
            "torch_version": "2.12.1+cu130",
            "diffusers_version": "0.39.0",
        }

    monkeypatch.setattr("wan_backend.probe_python_runtime", fake_probe)
    monkeypatch.setenv("DUCKMOTION_WAN_PYTHON", "/runtime/wan/python")
    readiness = backend.readiness(descriptor)

    assert readiness["ready"] is True
    assert readiness["cuda_available"] is True
    assert readiness["gpu_name"] == "NVIDIA GeForce RTX 5070 Ti"
    assert captured[0][0] == "/runtime/wan/python"
    assert ("diffusers", "WanPipeline") in captured[0][1]
    assert ("diffusers", "WanImageToVideoPipeline") in captured[0][1]


def test_resolver_readiness_preserves_backend_diagnostics(monkeypatch):
    descriptor = describe_video_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    backend = WanDiffusersBackend()
    monkeypatch.setattr(
        "wan_backend.probe_python_runtime",
        lambda *_args, **_kwargs: {
            "ready": True,
            "reason": None,
            "cuda_available": True,
            "gpu_name": "GPU",
        },
    )
    resolver = VideoBackendResolver()
    resolver.register(backend)

    result = resolver.readiness(descriptor)
    assert result["ready"] is True
    assert result["gpu_name"] == "GPU"


def test_wan_readiness_is_cached(monkeypatch):
    descriptor = describe_video_model("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    backend = WanDiffusersBackend()
    calls = []

    def fake_probe(*args, **kwargs):
        calls.append((args, kwargs))
        return {"ready": True, "reason": None, "cuda_available": True}

    monkeypatch.setattr("wan_backend.probe_python_runtime", fake_probe)
    first = backend.readiness(descriptor)
    second = backend.readiness(descriptor)

    assert second == first
    assert len(calls) == 1


def test_ltx_readiness_probes_two_stage_runtime_and_caches_result(monkeypatch):
    descriptor = describe_video_model("Lightricks/LTX-2.5-Diffusers")
    backend = LTX25IsolatedBackend()
    calls = []

    def fake_probe(python, symbols, **kwargs):
        calls.append((python, tuple(symbols), kwargs))
        return {"ready": True, "reason": None, "cuda_available": True}

    monkeypatch.setattr("ltx_backend.probe_python_runtime", fake_probe)

    first = backend.readiness(descriptor)
    second = backend.readiness(descriptor)

    assert second == first
    assert len(calls) == 1
    symbols = calls[0][1]
    assert ("diffusers", "LTX2LatentUpsamplePipeline") in symbols
    assert ("diffusers.pipelines.ltx2.latent_upsampler", "LTX2LatentUpsamplerModel") in symbols
    assert ("diffusers.pipelines.ltx2.utils", "STAGE_2_DISTILLED_SIGMA_VALUES") in symbols


def test_runtime_requirements_are_reproducible():
    root = Path(__file__).resolve().parents[1] / "runtime_requirements"
    wan = (root / "wan.txt").read_text(encoding="utf-8")
    ltx = (root / "ltx25.txt").read_text(encoding="utf-8")

    assert "diffusers==0.39.0" in wan
    assert "gguf==0.19.0" in wan
    assert "git+https://github.com/huggingface/diffusers.git@" in ltx
    assert "git+https://github.com/huggingface/diffusers.git\n" not in ltx
