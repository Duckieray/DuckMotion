from __future__ import annotations

import inspect

import plugin_backend


def _endpoint(router, path, method):
    method = method.upper()
    for route in router.routes:
        if getattr(route, "path", None) != path:
            continue
        if method in {str(value).upper() for value in (getattr(route, "methods", None) or set())}:
            return route.endpoint
    raise AssertionError(f"Missing route {method} {path}")


def test_router_is_composed_from_generic_runtime_and_storage_surfaces(monkeypatch):
    monkeypatch.setattr(plugin_backend.services, "load_config", lambda: {})
    monkeypatch.setattr(plugin_backend.runtime_surfaces, "health", lambda: {"mode": "model-driven-local-runtime"})
    monkeypatch.setattr(plugin_backend.runtime_surfaces, "get_config", lambda: {"config": {"model_id_or_path": ""}})
    monkeypatch.setattr(plugin_backend.runtime_surfaces, "engine_status", lambda: {"type": "model-driven-video-runtime"})
    monkeypatch.setattr(plugin_backend.runtime_surfaces, "unload", lambda: {"ok": True})
    monkeypatch.setattr(plugin_backend.services.storage, "list_jobs", lambda limit=50: [{"job_id": "dm_generic"}])
    monkeypatch.setattr(plugin_backend.services.storage, "list_staging", lambda limit=100: [{"name": "input.png"}])
    monkeypatch.setattr(plugin_backend.services.storage, "scan_gallery", lambda config, limit=100: [{"run_id": "generic-run"}])
    monkeypatch.setattr(plugin_backend, "discover_video_models", lambda _config: {
        "items": [{"name": "LTX-2.5", "source": "/cache/ltx", "supported": True, "capabilities": {"text_to_video": True, "image_to_video": True, "audio_output": True}}],
        "count": 1,
        "hf_cache": "/cache/hub",
    })

    router = plugin_backend.get_router()
    assert _endpoint(router, "/health", "GET")() == {"mode": "model-driven-local-runtime"}
    assert _endpoint(router, "/config", "GET")() == {"config": {"model_id_or_path": ""}}
    assert _endpoint(router, "/engine/status", "GET")() == {"type": "model-driven-video-runtime"}
    assert _endpoint(router, "/engine/unload", "POST")() == {"ok": True}
    assert _endpoint(router, "/engine/jobs", "GET")()["jobs"][0]["job_id"] == "dm_generic"
    assert _endpoint(router, "/staging", "GET")()["items"][0]["name"] == "input.png"
    assert _endpoint(router, "/gallery", "GET")()["items"][0]["run_id"] == "generic-run"
    payload = _endpoint(router, "/models", "GET")()
    assert payload["items"][0]["name"] == "LTX-2.5"
    assert payload["items"][0]["supported"] is True


def test_runtime_readiness_lists_discovered_models_without_loading_weights(monkeypatch):
    monkeypatch.setattr(plugin_backend.services, "load_config", lambda: {})
    monkeypatch.setattr(plugin_backend, "_register_installed_backends", lambda: None)
    monkeypatch.setattr(plugin_backend, "discover_video_models", lambda _config: {
        "items": [
            {
                "name": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "source": "/cache/wan/revision",
                "repo_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "location": "hf_cache",
            },
            {
                "name": "Lightricks/LTX-2.5-Diffusers",
                "source": "/cache/ltx/revision",
                "repo_id": "Lightricks/LTX-2.5-Diffusers",
                "location": "hf_cache",
            },
        ],
        "count": 2,
        "hf_cache": "/cache/hub",
    })
    monkeypatch.setattr(
        plugin_backend.backend_resolver,
        "readiness",
        lambda descriptor: {
            "ready": True,
            "reason": None,
            "cuda_available": True,
            "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
            "runtime_model": descriptor.name,
        },
    )

    payload = _endpoint(plugin_backend.get_router(), "/runtime-readiness", "GET")()
    assert payload["count"] == 2
    assert payload["items"][0]["runtime"]["gpu_name"] == "NVIDIA GeForce RTX 5070 Ti"
    assert payload["items"][0]["capabilities"]["text_to_video"] is True
    assert payload["items"][1]["capabilities"]["audio_output"] is True
    assert "architecture" not in payload["items"][0]
    assert "backend" not in payload["items"][0]


def _exercise_generate(monkeypatch, model_source, payload):
    monkeypatch.setattr(plugin_backend.services, "load_config", lambda: {"model_id_or_path": model_source})
    monkeypatch.setattr(plugin_backend, "_register_installed_backends", lambda: None)
    submitted = {}

    class FakeRuntime:
        backend_id = "fake"

    monkeypatch.setattr(plugin_backend.backend_resolver, "resolve", lambda descriptor: FakeRuntime())

    def fake_submit(descriptor, request, config):
        submitted["descriptor"] = descriptor
        submitted["request"] = request
        submitted["config"] = config
        return {"job_id": "dm_generic", "status": "queued"}

    monkeypatch.setattr(plugin_backend.job_coordinator, "submit", fake_submit)
    result = _endpoint(plugin_backend.get_router(), "/engine/generate", "POST")(payload)
    return result, submitted


def test_ltx_text_to_video_uses_generic_job_coordinator(monkeypatch):
    result, submitted = _exercise_generate(monkeypatch, "Lightricks/LTX-2.5-Diffusers", plugin_backend.GeneratePayload(prompt="cinematic red fox"))
    assert result["ok"] is True
    assert submitted["request"]["image_path"] is None
    assert submitted["descriptor"].backend == "ltx25_isolated"
    assert submitted["descriptor"].capabilities.audio_output is True


def test_generic_i2v_stability_hint_reaches_job_coordinator(monkeypatch):
    result, submitted = _exercise_generate(
        monkeypatch,
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        plugin_backend.GeneratePayload(
            image_path="/tmp/source.png",
            prompt="subtle movement",
            i2v_stability="locked",
        ),
    )
    assert result["ok"] is True
    assert submitted["request"]["image_path"] == "/tmp/source.png"
    assert submitted["request"]["i2v_stability"] == "locked"


def test_wan_i2v_uses_same_generic_job_coordinator(monkeypatch):
    result, submitted = _exercise_generate(monkeypatch, "Wan-AI/Wan2.2-I2V-A14B-Diffusers", plugin_backend.GeneratePayload(image_path="/tmp/source.png", prompt="move"))
    assert result["ok"] is True
    assert submitted["descriptor"].backend == "wan_diffusers"
    assert submitted["request"]["image_path"] == "/tmp/source.png"


def test_wan_ti2v_diffusers_can_submit_text_only_without_source_image(monkeypatch):
    result, submitted = _exercise_generate(monkeypatch, "Wan-AI/Wan2.2-TI2V-5B-Diffusers", plugin_backend.GeneratePayload(prompt="a cinematic rainy city"))
    assert result["ok"] is True
    assert submitted["descriptor"].capabilities.text_to_video is True
    assert submitted["descriptor"].capabilities.image_to_video is False
    assert submitted["request"]["image_path"] is None


def test_plugin_has_no_legacy_backend_or_family_dispatch():
    source = inspect.getsource(plugin_backend)
    assert "wan_impl" not in source
    assert "importlib" not in source
    assert "backend.py" not in source
    assert 'descriptor.backend == "ltx' not in source
    assert 'descriptor.backend == "wan' not in source
