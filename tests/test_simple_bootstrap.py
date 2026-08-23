from __future__ import annotations

import importlib.util
from pathlib import Path

from ltx_convrot_assets import is_ltx25_convrot_path
from runtime_paths import (
    RUNTIME_ENV_VARS,
    configure_default_runtime_env,
    default_runtime_python,
    resolve_runtime_python,
)


ROOT = Path(__file__).resolve().parents[1]


def test_default_runtime_paths_need_no_backend_exports(monkeypatch, tmp_path):
    monkeypatch.setenv("DUCKMOTION_RUNTIME_HOME", str(tmp_path / "runtimes"))
    for env_var in RUNTIME_ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)

    configured = configure_default_runtime_env()

    for runtime, env_var in RUNTIME_ENV_VARS.items():
        expected = str(default_runtime_python(runtime))
        assert configured[env_var] == expected
        assert resolve_runtime_python(runtime) == expected


def test_explicit_runtime_override_still_wins(monkeypatch, tmp_path):
    override = tmp_path / "custom-python"
    monkeypatch.setenv("DUCKMOTION_WAN_PYTHON", str(override))
    assert resolve_runtime_python("wan") == str(override)


def test_redgraft_ltx_name_is_convrot_candidate_without_rename(tmp_path):
    checkpoint = tmp_path / "redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors"
    assert is_ltx25_convrot_path(checkpoint) is True


def test_unrelated_ltx_safetensors_is_not_assumed_convrot(tmp_path):
    checkpoint = tmp_path / "ltx25-normal.safetensors"
    assert is_ltx25_convrot_path(checkpoint) is False


def test_plugin_startup_configures_default_runtime_environment():
    source = (ROOT / "plugin_backend.py").read_text(encoding="utf-8")
    assert "configure_default_runtime_env()" in source


def test_setup_and_doctor_are_first_class_commands():
    setup = ROOT / "tools" / "setup.py"
    doctor = ROOT / "tools" / "doctor.py"
    assert setup.exists()
    assert doctor.exists()

    setup_source = setup.read_text(encoding="utf-8")
    doctor_source = doctor.read_text(encoding="utf-8")
    assert "prepare_model_runtimes.py" in setup_source
    assert "install_webbduck_plugin.py" in setup_source
    assert "discover_video_models" in doctor_source
    assert "backend_resolver.readiness" in doctor_source
