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


def test_repair_only_repairs_convrot_owned_checkout_without_reinstall(monkeypatch, tmp_path):
    runtime_root = tmp_path / "runtimes"
    python_path = runtime_root / "ltx25_convrot" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.touch()

    calls: list[tuple[Path, bool]] = []

    def fake_prepare_comfy(env_root: Path, *, dry_run: bool):
        calls.append((env_root, dry_run))
        return env_root / "comfyui"

    monkeypatch.setattr(prep, "prepare_comfy_checkout", fake_prepare_comfy)

    env_var, repaired_python = prep.repair_runtime(
        "ltx25_convrot",
        root=runtime_root,
        dry_run=False,
    )

    assert env_var == "DUCKMOTION_LTX_CONVROT_PYTHON"
    assert repaired_python == python_path
    assert calls == [(runtime_root / "ltx25_convrot", False)]


def test_repair_only_does_not_invent_missing_runtime(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        prep.repair_runtime("ltx25_convrot", root=tmp_path / "runtimes", dry_run=False)


def test_setup_skip_runtimes_still_uses_repair_only():
    source = (ROOT / "tools" / "setup.py").read_text(encoding="utf-8")
    assert '"--repair-only"' in source
