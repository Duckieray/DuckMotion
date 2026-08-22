from __future__ import annotations

from fastapi import APIRouter

import plugin_backend


def _fake_wan_router():
    router = APIRouter()

    @router.get("/models/discover")
    def old_discovery():
        return {"items": [{"path": "/wan"}]}

    @router.post("/engine/generate")
    def old_generate(payload):
        return {"delegated": True, "image_path": payload.image_path}

    @router.get("/health")
    def health():
        return {"ok": True}

    return router


def _endpoint(router, path, method):
    method = method.upper()
    for route in router.routes:
        if getattr(route, "path", None) != path:
            continue
        if method in {str(value).upper() for value in (getattr(route, "methods", None) or set())}:
            return route.endpoint
    raise AssertionError(f"Missing route {method} {path}")


def test_router_exposes_generic_model_catalog(monkeypatch):
    monkeypatch.setattr(plugin_backend.wan, "get_router", lambda _manifest=None: _fake_wan_router())
    monkeypatch.setattr(plugin_backend.wan, "_load_config", lambda: {})
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
    assert _endpoint(router, "/health", "GET")() == {"ok": True}
    payload = _endpoint(router, "/models", "GET")()
    assert payload["items"][0]["name"] == "LTX-2.5"
    assert payload["items"][0]["supported"] is True
    assert callable(_endpoint(router, "/models/discover", "GET"))


def test_ltx_text_to_video_does_not_require_source_image(monkeypatch):
    monkeypatch.setattr(plugin_backend.wan, "get_router", lambda _manifest=None: _fake_wan_router())
    monkeypatch.setattr(plugin_backend.wan, "_load_config", lambda: {
        "model_id_or_path": "Lightricks/LTX-2.5-Diffusers"
    })
    submitted = {}

    def fake_submit(payload, config, descriptor):
        submitted["payload"] = payload
        submitted["descriptor"] = descriptor
        return {"job_id": "dm_ltx", "status": "queued"}

    monkeypatch.setattr(plugin_backend, "_submit_ltx_job", fake_submit)
    router = plugin_backend.get_router()
    endpoint = _endpoint(router, "/engine/generate", "POST")
    result = endpoint(plugin_backend.GeneratePayload(prompt="cinematic red fox"))

    assert result["ok"] is True
    assert result["job"]["job_id"] == "dm_ltx"
    assert submitted["payload"].image_path is None
    assert submitted["descriptor"].capabilities.audio_output is True


def test_wan_still_delegates_to_current_internal_implementation(monkeypatch):
    monkeypatch.setattr(plugin_backend.wan, "get_router", lambda _manifest=None: _fake_wan_router())
    monkeypatch.setattr(plugin_backend.wan, "_load_config", lambda: {
        "model_id_or_path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    })
    router = plugin_backend.get_router()
    endpoint = _endpoint(router, "/engine/generate", "POST")
    result = endpoint(plugin_backend.GeneratePayload(image_path="/tmp/source.png", prompt="move"))
    assert result["delegated"] is True
    assert result["image_path"] == "/tmp/source.png"


def test_ltx_constraints_snap_to_model_requirements():
    descriptor = plugin_backend.describe_video_model("Lightricks/LTX-2.5-Diffusers")
    payload = plugin_backend.GeneratePayload(
        prompt="test",
        width=777,
        height=530,
        num_frames=120,
        fps=24,
    )
    params = plugin_backend._ltx_params(payload, descriptor)
    assert params["width"] % 32 == 0
    assert params["height"] % 32 == 0
    assert params["num_frames"] % 8 == 1
    assert params["guidance_scale"] == 1.0
