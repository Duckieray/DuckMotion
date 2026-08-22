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


def test_discovery_keeps_legacy_shape_and_makes_catalog_picker_visible(monkeypatch):
    monkeypatch.setattr(plugin_backend.legacy, "_discover_local_models", lambda _config: {
        "items": [{"path": "/wan", "label": "Wan", "source": "checkpoint_wan"}],
        "gguf_candidates": [],
    })
    monkeypatch.setattr(plugin_backend, "discover_video_models", lambda _config: {
        "items": [
            {
                "name": "Wan",
                "source": "/wan",
                "location": "local",
                "supported": True,
                "capabilities": {"image_to_video": True},
            },
            {
                "name": "LTX-2.5",
                "source": "/cache/ltx",
                "location": "hf_cache",
                "supported": False,
                "capabilities": {"text_to_video": True, "image_to_video": True, "audio_output": True},
                "constraints": {"dimension_multiple": 32},
            },
        ],
        "count": 2,
        "hf_cache": "/cache/hub",
    })

    payload = plugin_backend._catalog_with_legacy_compat({})

    assert len(payload["items"]) == 2
    wan = next(row for row in payload["items"] if row["path"] == "/wan")
    ltx = next(row for row in payload["items"] if row["path"] == "/cache/ltx")

    assert wan["label"] == "Wan"
    assert wan["capabilities"]["image_to_video"] is True
    assert ltx["label"] == "LTX-2.5 — runtime unavailable"
    assert ltx["source"] == "hf_cache"
    assert ltx["supported"] is False
    assert ltx["capabilities"]["audio_output"] is True
    assert payload["catalog_count"] == 2


def test_picker_adapter_does_not_expose_architecture_or_backend():
    row = plugin_backend._picker_item_from_catalog({
        "name": "LTX-2.5",
        "source": "/cache/ltx",
        "location": "hf_cache",
        "supported": False,
        "capabilities": {"text_to_video": True},
        "architecture": "ltx25",
        "backend": "ltx25_isolated",
    })
    assert row is not None
    assert "architecture" not in row
    assert "backend" not in row
    assert row["label"].endswith("runtime unavailable")


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
