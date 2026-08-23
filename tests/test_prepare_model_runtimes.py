from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "duckmotion_prepare_model_runtimes",
    ROOT / "tools" / "prepare_model_runtimes.py",
)
assert SPEC is not None and SPEC.loader is not None
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


def test_standard_runtimes_keep_newer_torch_stack():
    assert prep.resolve_torch_stack("wan") == ("2.12.1", "0.27.1", None)
    assert prep.resolve_torch_stack("ltx25") == ("2.12.1", "0.27.1", None)


def test_convrot_uses_official_matched_cu130_audio_stack():
    assert prep.resolve_torch_stack("ltx25_convrot") == (
        "2.11.0",
        "0.26.0",
        "2.11.0",
    )


def test_convrot_rejects_partial_torch_stack_overrides():
    with pytest.raises(ValueError, match="matched torch/torchvision/torchaudio"):
        prep.resolve_torch_stack("ltx25_convrot", torch_version="2.12.1")

    with pytest.raises(ValueError, match="matched torch/torchvision/torchaudio"):
        prep.resolve_torch_stack(
            "ltx25_convrot",
            torch_version="2.11.0",
            torchvision_version="0.26.0",
        )


def test_convrot_allows_explicit_matched_override():
    assert prep.resolve_torch_stack(
        "ltx25_convrot",
        torch_version="2.10.0",
        torchvision_version="0.25.0",
        torchaudio_version="2.10.0",
    ) == ("2.10.0", "0.25.0", "2.10.0")
