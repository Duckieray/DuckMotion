from __future__ import annotations

import os
from pathlib import Path

import pytest

from ltx_convrot_backend import LTX25ConvRotBackend
from runtime_paths import (
    CONVROT_COMFY_ROOT_ENV,
    configure_default_runtime_env,
    resolve_convrot_comfy_root,
    runtime_root_from_python,
)


def _make_python_symlink(tmp_path: Path) -> tuple[Path, Path]:
    base_python = tmp_path / "base" / "bin" / "python"
    base_python.parent.mkdir(parents=True)
    base_python.write_text("#!/bin/sh\n", encoding="utf-8")

    runtime_root = tmp_path / "runtimes" / "ltx25_convrot"
    runtime_python = runtime_root / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    try:
        runtime_python.symlink_to(base_python)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable on this platform: {exc}")
    return runtime_root, runtime_python


def test_runtime_root_from_python_does_not_follow_venv_symlink(tmp_path: Path):
    runtime_root, runtime_python = _make_python_symlink(tmp_path)

    assert runtime_python.resolve() != runtime_python
    assert runtime_root_from_python(runtime_python) == runtime_root


def test_default_convrot_comfy_root_stays_beside_overridden_venv(monkeypatch, tmp_path: Path):
    runtime_root, runtime_python = _make_python_symlink(tmp_path)
    monkeypatch.setenv("DUCKMOTION_LTX_CONVROT_PYTHON", str(runtime_python))
    monkeypatch.delenv(CONVROT_COMFY_ROOT_ENV, raising=False)

    expected = runtime_root / "comfyui"
    assert resolve_convrot_comfy_root() == expected

    configured = configure_default_runtime_env()
    assert configured[CONVROT_COMFY_ROOT_ENV] == str(expected)
    assert os.environ[CONVROT_COMFY_ROOT_ENV] == str(expected)
    assert LTX25ConvRotBackend._comfy_root(str(runtime_python)) == expected


def test_explicit_convrot_comfy_root_override_still_wins(monkeypatch, tmp_path: Path):
    explicit = tmp_path / "custom-comfy"
    monkeypatch.setenv(CONVROT_COMFY_ROOT_ENV, str(explicit))

    assert resolve_convrot_comfy_root() == explicit
    assert LTX25ConvRotBackend._comfy_root("/irrelevant/bin/python") == explicit
