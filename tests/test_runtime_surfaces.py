from __future__ import annotations

import json
from pathlib import Path

from model_runtime import VideoBackend, VideoBackendResolver, describe_video_model
from runtime_services import VideoRuntimeServices
from runtime_surfaces import VideoConfigPayload, VideoRuntimeSurfaces
from storage_runtime import VideoStorageRuntime


class _Backend(VideoBackend):
    backend_id = "ltx25_isolated"

    def can_handle(self, descriptor):
        return descriptor.backend == self.backend_id

    def generate(self, descriptor, request, **kwargs):
        return {}


class _Services:
    def __init__(self, tmp_path: Path):
        self.config = {
            "model_id_or_path": "Lightricks/LTX-2.5-Diffusers",
            "models_dir": "/models",
            "output_dir": str(tmp_path / "videos"),
            "runtime_backend": "wan-shaped-internal-value",
            "gguf_transformer_path": "/should/not/leak.gguf",
        }
        self.saved = None
        self.tmp_path = tmp_path

    def load_config(self):
        return dict(self.config)

    def save_config(self, config):
        self.saved = dict(config)
        self.config.update(config)

    def resolve_output_dir(self, config):
        path = Path(config.get("output_dir") or self.tmp_path / "videos")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def runtime_profile_safe(self):
        return {"ok": True, "profile": {"device": "cuda", "dtype": "bfloat16"}}

    def output_writable(self, config):
        return True, None

    def gpu_lease(self):
        return {"held": False}

    def queue_snapshot(self):
        return {"queued": 0, "running": 0, "cancel_requested": 0}


def _surfaces(tmp_path: Path):
    services = _Services(tmp_path)
    resolver = VideoBackendResolver()

    def register():
        if "ltx25_isolated" not in resolver.ids():
            resolver.register(_Backend())

    surfaces = VideoRuntimeSurfaces(
        services=services,
        resolver=resolver,
        describe_model=describe_video_model,
        discover_models=lambda _config: {"items": [], "count": 2, "hf_cache": "/cache/hub"},
        register_backends=register,
    )
    return surfaces, services


def test_health_is_model_driven_and_hides_backend_family_details(tmp_path):
    surfaces, _services = _surfaces(tmp_path)

    payload = surfaces.health()

    assert payload["mode"] == "model-driven-local-runtime"
    assert payload["selected_model"]["capabilities"]["text_to_video"] is True
    assert payload["ready"]["engine_ready"] is True
    assert payload["catalog"]["count"] == 2
    assert "architecture" not in payload["selected_model"]
    assert "backend" not in payload["selected_model"]
    assert "installation" not in payload
    assert "runtime_plan" not in json.dumps(payload)
    assert "gguf" not in json.dumps(payload).lower()
    assert "wan-shaped-internal-value" not in json.dumps(payload)


def test_public_config_contains_only_model_neutral_fields(tmp_path):
    surfaces, _services = _surfaces(tmp_path)

    config = surfaces.get_config()["config"]

    assert set(config) == {
        "model_id_or_path",
        "models_dir",
        "output_dir",
        "resolved_output_dir",
    }


def test_config_update_does_not_require_generation_defaults(tmp_path):
    surfaces, services = _surfaces(tmp_path)

    result = surfaces.set_config(
        VideoConfigPayload(
            model_id_or_path="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            output_dir=str(tmp_path / "new"),
        )
    )

    assert result["ok"] is True
    assert services.saved["model_id_or_path"] == "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    assert "default_frames" not in result["config"]
    assert "runtime_backend" not in result["config"]


def test_engine_status_contains_generic_runtime_state(tmp_path):
    surfaces, _services = _surfaces(tmp_path)

    status = surfaces.engine_status()

    assert status["type"] == "model-driven-video-runtime"
    assert status["ready"] is True
    assert status["queue"]["running"] == 0
    assert "pipeline_class" not in json.dumps(status)
    assert "backend" not in status["selected_model"]


def test_runtime_services_does_not_implicitly_select_wan_and_persists_generic_fields(tmp_path, monkeypatch):
    monkeypatch.delenv("DUCKMOTION_MODEL_ID_OR_PATH", raising=False)
    storage = VideoStorageRuntime(
        config_file=tmp_path / "duckmotion_config.json",
        jobs_file=tmp_path / "jobs.json",
        staging_dir=tmp_path / "staging",
        default_output_dir=tmp_path / "videos",
    )
    services = VideoRuntimeServices(storage=storage)

    loaded = services.load_config()
    assert loaded["model_id_or_path"] == ""

    loaded["model_id_or_path"] = "Lightricks/LTX-2.5-Diffusers"
    loaded["models_dir"] = "/models"
    loaded["output_dir"] = "/videos"
    loaded["runtime_backend"] = "internal-only"
    services.save_config(loaded)

    saved = json.loads((tmp_path / "duckmotion_config.json").read_text(encoding="utf-8"))
    assert saved == {
        "model_id_or_path": "Lightricks/LTX-2.5-Diffusers",
        "models_dir": "/models",
        "output_dir": "/videos",
    }


def test_runtime_services_has_no_backend_implementation_object():
    import inspect

    source = inspect.getsource(VideoRuntimeServices)
    assert "implementation" not in source.lower()
    assert "_impl" not in source
    assert "wan" not in source.lower()
