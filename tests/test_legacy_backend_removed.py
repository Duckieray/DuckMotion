from __future__ import annotations

import inspect
from pathlib import Path

import host_runtime
import plugin_backend
import runtime_services
import wan_backend


def test_legacy_backend_module_is_deleted():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "backend.py").exists()


def test_plugin_composition_does_not_load_a_wan_implementation_module():
    source = inspect.getsource(plugin_backend)
    assert "importlib" not in source
    assert "wan_impl" not in source
    assert "backend.py" not in source
    assert "VideoRuntimeServices()" in source
    assert "ensure_wan_registered()" in source


def test_runtime_services_only_compose_storage_and_host_runtime():
    source = inspect.getsource(runtime_services.VideoRuntimeServices)
    assert "_impl" not in source
    assert "implementation" not in source.lower()
    assert "host_runtime" in source
    assert "storage" in source


def test_host_runtime_is_architecture_neutral():
    source = inspect.getsource(host_runtime).lower()
    assert "wan" not in source
    assert "ltx" not in source
    assert "diffusers" not in source


def test_wan_backend_is_process_isolated_and_has_no_legacy_hooks():
    source = inspect.getsource(wan_backend.WanDiffusersBackend)
    assert "subprocess.Popen" in source
    assert "_prepare_runtime_for_wan" not in source
    assert "_generate_frames_with_diffusers" not in source
    assert "_write_video_outputs" not in source
