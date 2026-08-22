from __future__ import annotations

from pathlib import Path

import wan_backend
from model_runtime import describe_video_model
from wan_backend import WanDiffusersBackend
from wan_worker import _gguf_pair_path, _gguf_role


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "checkpoint" / "wan" / "Wan2.2-Enhanced-NSFW-I2V-T2V"
    root.mkdir(parents=True)
    high = root / "Wan2.2_Enhanced_NSFW_I2V_T2V_Q8_H.gguf"
    low = root / "Wan2.2_Enhanced_NSFW_I2V_T2V_Q8_L.gguf"
    high.write_bytes(b"gguf-high")
    low.write_bytes(b"gguf-low")
    return high, low


def test_gguf_descriptor_stays_on_generic_wan_backend(tmp_path):
    high, _low = _pair(tmp_path)
    descriptor = describe_video_model(
        str(high),
        name="Wan2.2 Enhanced NSFW I2V T2V Q8",
    )
    assert descriptor.architecture == "wan22"
    assert descriptor.backend == "wan_diffusers"
    assert descriptor.detection["format"] == "gguf"
    assert descriptor.detection["pair_role"] == "H"
    assert descriptor.capabilities.text_to_video is True
    assert descriptor.capabilities.image_to_video is True
    assert descriptor.capabilities.source_image_required is False
    assert descriptor.supported is True

    public = descriptor.to_public_dict()
    assert "architecture" not in public
    assert "backend" not in public
    assert "format" not in public


def test_gguf_pair_resolution_finds_both_denoisers(tmp_path):
    high, low = _pair(tmp_path)
    assert _gguf_role(high) == "H"
    assert _gguf_role(low) == "L"
    assert _gguf_pair_path(high) == low
    assert _gguf_pair_path(low) == high


def test_gguf_readiness_probes_quantized_runtime_requirements(tmp_path, monkeypatch):
    high, _low = _pair(tmp_path)
    descriptor = describe_video_model(
        str(high),
        name="Wan2.2 Enhanced NSFW I2V T2V Q8",
    )
    captured = {}

    def fake_probe(python_exe, symbols, **_kwargs):
        captured["python"] = python_exe
        captured["symbols"] = tuple(symbols)
        return {"ready": True, "cuda_available": True}

    monkeypatch.setattr(wan_backend, "probe_python_runtime", fake_probe)
    monkeypatch.setenv("DUCKMOTION_WAN_PYTHON", "/runtime/wan/python")
    result = WanDiffusersBackend().readiness(descriptor)

    assert result["ready"] is True
    assert result["source_format"] == "gguf"
    assert captured["python"] == "/runtime/wan/python"
    assert ("diffusers", "WanTransformer3DModel") in captured["symbols"]
    assert ("diffusers", "GGUFQuantizationConfig") in captured["symbols"]
    assert ("gguf", "GGUFReader") in captured["symbols"]


def test_worker_injects_both_gguf_transformers_into_pipeline():
    root = Path(__file__).resolve().parents[1]
    text = (root / "wan_worker.py").read_text(encoding="utf-8")
    assert "WanTransformer3DModel.from_single_file" in text
    assert "GGUFQuantizationConfig(compute_dtype=dtype)" in text
    assert 'kwargs["transformer_2"] = transformer_2' in text
    assert "DUCKMOTION_WAN_GGUF_I2V_BASE" in text
    assert "DUCKMOTION_WAN_GGUF_T2V_BASE" in text
