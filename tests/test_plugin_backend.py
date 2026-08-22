from __future__ import annotations

from fastapi import APIRouter

import plugin_backend


def _fake_implementation_router():
    router = APIRouter()

    @router.get("/models/discover")
    def old_discovery():
        return {"items": [{"path": "/wan"}]}

    @router.post("/engine/generate")
    def old_generate(payload):
        return {"legacy": True}

    @router.get("/health")
    def old_health():
        return {"legacy_health": True}

    @router.get("/config")
    def old_config():
        return {"runtime_backend": "wan"}

    @router.get("/engine/status")
    def old_status():
        return {"pipeline": "WanImageToVideoPipeline"}

    @router.post("/engine/unload")
    def old_unload():
        return {"legacy_unload": True}

    @router.get("/gallery")
    def gallery():
        return {"items": []}

    return router


def _endpoint(router, path, method):
    method = method.upper()
    for route in router.routes:
        if getattr(route, "path", None) != path:
            continue
        if method in {str(value).upper() for value in (getattr(route, "methods", None) or set())}:
            return route.endpoint
    raise AssertionError(f"Missing route {method} {path}")


def test_router_replaces_model_config_health_and_status_surfaces(monkeypatch):
    monkeypatch.setattr(plugin_backend.wan_impl, "get_router", lambda _manifest=None: _fake_implementation_router())
    monkeypatch.setattr(plugin_backend.services, "load_config", lambda: {})
    monkeypatch.setattr(plugin_backend.runtime_surfaces, "health", lambda: {"mode": "model-driven-local-runtime"})
    monkeypatch.setattr(plugin_backend.runtime_surfaces, "get_config", lambda: {"config": {"model_id_or_path": ""}})
    monkeypatch.setattr(plugin_backend.runtime_surfaces, "engine_status", lambda: {"type": "model-driven-video-runtime"})
    monkeypatch.setattr(plugin_backend.runtime_surfaces, "unload", lambda: {"ok": True})
    monkeypatch.setattr(plugin_backend, "discover_video_models", lambda _config: {
        "items": [
            {
                "name": "LTX-2.5",
                "source": "/cache/ltx",
                "supported": True,
                "capabilities": {"text_to_video": True, "image_to_video": True, "audio_output": True},
            }
        ],
        "count": 1,
        "hf_cache": "/cache/hub",
    })

    router = plugin_backend.get_router()
    assert _endpoint(router, "/health", "GET")() == {"mode": "model-driven-local-runtime"}
    assert _endpoint(router, "/config", "GET")() == {"config": {"model_id_or_path": ""}}
    assert _endpoint(router, "/engine/status", "GET")() == {"type": "model-driven-video-runtime"}
    assert _endpoint(router, "/engine/unload", "POST")() == {"ok": True}
    assert _endpoint(router, "/gallery", "GET")() == {"items": []}

    payload = _endpoint(router, "/models", "GET")()
    assert payload["items"][0]["name"] == "LTX-2.5"
    assert payload["items"][0]["supported"] is True
    assert callable(_endpoint(router, "/models/discover", "GET"))


def _exercise_generate(monkeypatch, model_source, payload):
    monkeypatch.setattr(plugin_backend.wan_impl, "get_router", lambda _manifest=None: _fake_implementation_router())
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
    router = plugin_backend.get_router()
    result = _endpoint(router, "/engine/generate", "POST")(payload)
    return result, submitted


def test_ltx_text_to_video_uses_generic_job_coordinator(monkeypatch):
    result, submitted = _exercise_generate(
        monkeypatch,
        "Lightricks/LTX-2.5-Diffusers",
        plugin_backend.GeneratePayload(prompt="cinematic red fox"),
    )
    assert result["ok"] is True
    assert result["job"]["job_id"] == "dm_generic"
    assert submitted["request"]["image_path"] is None
    assert submitted["descriptor"].backend == "ltx25_isolated"
    assert submitted["descriptor"].capabilities.audio_output is True


def test_wan_uses_same_generic_job_coordinator(monkeypatch):
    result, submitted = _exercise_generate(
        monkeypatch,
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        plugin_backend.GeneratePayload(image_path="/tmp/source.png", prompt="move"),
    )
    assert result["ok"] is True
    assert result["job"]["job_id"] == "dm_generic"
    assert submitted["descriptor"].backend == "wan_diffusers"
    assert submitted["request"]["image_path"] == "/tmp/source.png"


def test_route_contains_no_backend_family_dispatch_or_legacy_config_read():
    import inspect

    source = inspect.getsource(plugin_backend.get_router)
    assert 'descriptor.backend == "ltx' not in source
    assert 'descriptor.backend == "wan' not in source
    assert "wan_impl._load_config" not in source
