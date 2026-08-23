from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from ltx_convrot_assets import is_ltx25_convrot_path
from model_runtime import describe_video_model
from runtime_paths import (
    RUNTIME_ENV_VARS,
    configure_default_runtime_env,
    default_runtime_python,
    resolve_runtime_python,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location(
        "duckmotion_run_hardware_smoke",
        ROOT / "tools" / "run_hardware_smoke.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _smoke_args(**overrides):
    values = {
        "wan_5b_model": None,
        "wan_i2v_model": None,
        "ltx_model": None,
        "ltx_convrot_model": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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


def test_redgraft_descriptor_routes_to_convrot_backend_without_rename(tmp_path):
    checkpoint = tmp_path / "redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors"
    checkpoint.write_bytes(b"")

    descriptor = describe_video_model(str(checkpoint))

    assert descriptor.architecture == "ltx25"
    assert descriptor.backend == "ltx25_convrot"
    assert descriptor.detection["format"] == "int8_convrot"
    assert descriptor.detection["variant"] == "convrot"
    assert descriptor.defaults["width"] == 1152
    assert descriptor.defaults["height"] == 768
    assert descriptor.defaults["num_frames"] == 241
    assert descriptor.constraints["checkpoint_recipe_required"] is True


def test_unrelated_ltx_safetensors_is_not_assumed_convrot(tmp_path):
    checkpoint = tmp_path / "ltx25-normal.safetensors"
    assert is_ltx25_convrot_path(checkpoint) is False


def test_smoke_runner_auto_matches_standard_ltx_and_redgraft():
    smoke = _load_smoke_module()
    standard = {
        "name": "Lightricks/LTX-2.5-Diffusers",
        "source": "Lightricks/LTX-2.5-Diffusers",
        "capabilities": {"audio_output": True},
    }
    redgraft = {
        "name": "redgraftLTX25Fast2K_ltx25RedgraftNSFW",
        "source": "/models/redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors",
        "capabilities": {"audio_output": True},
    }

    rows = smoke._rows(_smoke_args(), [standard, redgraft], "/tmp/source.png")
    by_name = {row["row"]: row for row in rows}

    assert by_name["ltx25-t2v-canary"]["model"] is standard
    assert by_name["ltx25-i2v-canary"]["model"] is standard
    assert by_name["ltx25-convrot-t2v-canary"]["model"] is redgraft
    assert by_name["ltx25-convrot-i2v-canary"]["model"] is redgraft


def test_smoke_runner_keeps_exact_row_names_when_model_missing():
    smoke = _load_smoke_module()
    rows = smoke._rows(_smoke_args(), [], None)
    names = {row["row"] for row in rows}

    assert "ltx25-t2v-canary" in names
    assert "ltx25-i2v-canary" in names
    assert "ltx25-convrot-t2v-canary" in names
    assert "ltx25-convrot-i2v-canary" in names


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
