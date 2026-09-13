from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lora_runtime
import ltx_convrot_runtime_worker
import ltx_convrot_worker
import ltx_worker


def _make_lora_library(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    root = tmp_path / "lora"
    ltx = root / "ltx"
    ltx.mkdir(parents=True)
    omni = ltx / "OmniNFT.safetensors"
    omni.write_bytes(b"test")
    (root / "loras.json").write_text(
        '{"OmniNFT":{"weight":0.8,"trigger":"OMNINFT","description":"NFT motion adapter"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("WEBBDUCK_LORA_DIR", str(root))
    return root, omni


def test_discovers_ltx_namespace_from_webbduck_lora_root(tmp_path, monkeypatch):
    _root, omni = _make_lora_library(tmp_path, monkeypatch)

    items = lora_runtime.discover_ltx_loras()

    assert items == [
        {
            "name": "OmniNFT",
            "file": "ltx/OmniNFT.safetensors",
            "weight": 0.8,
            "trigger": "OMNINFT",
            "description": "NFT motion adapter",
        }
    ]
    normalized = lora_runtime.normalize_lora_selection(
        [{"name": "OmniNFT", "weight": 1.25}], supported=True
    )
    materialized = lora_runtime.materialize_lora_paths(normalized)
    assert materialized[0]["path"] == str(omni.resolve())
    assert materialized[0]["weight"] == 1.25


def test_lora_selection_rejects_unknown_and_outside_namespace(tmp_path, monkeypatch):
    root, _omni = _make_lora_library(tmp_path, monkeypatch)
    outside = root / "sdxl" / "other.safetensors"
    outside.parent.mkdir()
    outside.write_bytes(b"test")

    with pytest.raises(ValueError, match="Unknown LTX LoRA"):
        lora_runtime.normalize_lora_selection([{"name": "Other"}], supported=True)

    with pytest.raises(ValueError, match="does not support LoRAs"):
        lora_runtime.normalize_lora_selection([{"name": "OmniNFT"}], supported=False)

    with pytest.raises(ValueError, match="outside the LTX LoRA namespace"):
        lora_runtime.materialize_lora_paths(
            [{"name": "Other", "file": "sdxl/other.safetensors", "weight": 1.0}]
        )


def test_supports_only_installed_ltx_backends():
    assert lora_runtime.supports_loras(
        SimpleNamespace(supported=True, architecture="ltx25", backend="ltx25_isolated")
    )
    assert lora_runtime.supports_loras(
        SimpleNamespace(supported=True, architecture="ltx25", backend="ltx25_convrot")
    )
    assert not lora_runtime.supports_loras(
        SimpleNamespace(supported=True, architecture="wan22", backend="wan22_isolated")
    )


def test_diffusers_worker_loads_multiple_adapters(tmp_path):
    first = tmp_path / "one.safetensors"
    second = tmp_path / "two.safetensors"
    first.write_bytes(b"one")
    second.write_bytes(b"two")

    calls = []

    class FakePipe:
        def load_lora_weights(self, path, **kwargs):
            calls.append(("load", path, kwargs))

        def set_adapters(self, names, adapter_weights=None):
            calls.append(("set", names, adapter_weights))

    applied = ltx_worker._apply_loras(
        FakePipe(),
        [
            {"name": "One", "path": str(first), "weight": 0.6},
            {"name": "Two", "path": str(second), "weight": 1.2},
        ],
    )

    assert applied == [{"name": "One", "weight": 0.6}, {"name": "Two", "weight": 1.2}]
    assert calls[0][2]["weight_name"] == "one.safetensors"
    assert calls[1][2]["weight_name"] == "two.safetensors"
    assert calls[2] == ("set", ["duckmotion_lora_0", "duckmotion_lora_1"], [0.6, 1.2])


def test_convrot_dispatch_applies_lora_after_unet_load(tmp_path, monkeypatch):
    lora = tmp_path / "OmniNFT.safetensors"
    lora.write_bytes(b"test")
    calls = []

    def fake_call(_nodes, node_name, **kwargs):
        calls.append((node_name, kwargs))
        if node_name == "UNETLoader":
            return ("base-model",)
        if node_name == "LoraLoaderModelOnly":
            return (f"{kwargs['model']}+{kwargs['lora_name']}@{kwargs['strength_model']}",)
        return (None,)

    monkeypatch.setattr(ltx_convrot_worker, "_call_node", fake_call)
    monkeypatch.setattr(
        ltx_convrot_runtime_worker,
        "_ACTIVE_LORAS",
        [{"name": "OmniNFT", "path": str(lora), "weight": 0.75}],
    )

    ltx_convrot_runtime_worker._install_lora_dispatch()
    result = ltx_convrot_worker._call_node(object(), "UNETLoader", unet_name="model.safetensors")

    assert result == ("base-model+OmniNFT.safetensors@0.75",)
    assert [name for name, _kwargs in calls] == ["UNETLoader", "LoraLoaderModelOnly"]
