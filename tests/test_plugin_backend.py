from __future__ import annotations

from fastapi import APIRouter, HTTPException
import pytest

import plugin_backend


def _fake_legacy_router():
    router = APIRouter()

    @router.get("/models/discover")
    def old_discovery():
        return {"items": [{"path": "/wan"}]}

    @router.post("/engine/generate")
    def old_generate(payload):
        return {"delegated": True, "payload": payload}

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


def test_discovery_keeps_legacy_shape_and_adds_generic_catalog(monkeypatch):
    monkeypatch.setattr(plugin_backend.legacy, "_discover_local_models", lambda _config: {
        "items": [{"path": "/wan"}],
        "gguf_candidates": [],
    })
    monkeypatch.setattr(plugin_backend, "discover_video_models", lambda _config: {
        "items": [{"name": "LTX-2.5", "source": "/cache/ltx", "supported": False}],
        "count": 1,
        "hf_cache": "/cache/hub",
    })

    payload = plugin_backend._catalog_with_legacy_compat({})
    assert payload["items"] == [{"path": "/wan"}]
    assert payload["catalog_items"][0]["name"] == "LTX-2.5"
    assert payload["catalog_count"] == 1


def test_router_replaces_generate_and_discovery_but_preserves_other_routes(monkeypatch):
    monkeypatch.setattr(plugin_backend.legacy, "get_router", lambda _manifest=None: _fake_legacy_router())
    monkeypatch.setattr(plugin_backend.legacy, "_load_config", lambda: {
        "model_id_or_path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    })

    router = plugin_backend.get_router()
    assert _endpoint(router, "/health", "GET")() == {"ok": True}
    assert callable(_endpoint(router, "/models/discover", "GET"))
    assert callable(_endpoint(router, "/models/catalog", "GET"))
    assert callable(_endpoint(router, "/engine/generate", "POST"))


def test_generate_blocks_ltx_before_wan_delegate(monkeypatch):
    fake_router = _fake_legacy_router()
    monkeypatch.setattr(plugin_backend.legacy, "get_router", lambda _manifest=None: fake_router)
    monkeypatch.setattr(plugin_backend.legacy, "_load_config", lambda: {
        "model_id_or_path": "Lightricks/LTX-2.5-Diffusers"
    })

    router = plugin_backend.get_router()
    endpoint = _endpoint(router, "/engine/generate", "POST")

    class Payload:
        image_path = "/tmp/source.png"

    with pytest.raises(HTTPException) as exc_info:
        endpoint(Payload())
    assert exc_info.value.status_code == 409
    detail = exc_info.value.detail
    assert detail["error"] == "Model runtime unavailable."
    assert detail["model"]["capabilities"]["text_to_video"] is True
