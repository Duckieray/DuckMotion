"""Capability-aware DuckMotion plugin entrypoint.

The existing backend.py remains the proven Wan implementation. This wrapper
adds model-driven discovery and validates the configured model before delegating
to that legacy path. It deliberately prevents recognized-but-unimplemented
models such as LTX-2.5 from falling into the Wan loader.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable

from fastapi import APIRouter, HTTPException


PLUGIN_ROOT = Path(__file__).resolve().parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from model_discovery import discover_video_models
from model_runtime import describe_video_model


def _load_legacy_backend():
    spec = importlib.util.spec_from_file_location(
        "duckmotion_legacy_backend",
        PLUGIN_ROOT / "backend.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load DuckMotion legacy backend module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy_backend()


def _route_endpoint(router: APIRouter, path: str, method: str) -> Callable[..., Any] | None:
    method = method.upper()
    for route in router.routes:
        if getattr(route, "path", None) != path:
            continue
        methods = {str(value).upper() for value in (getattr(route, "methods", None) or set())}
        if method in methods:
            return getattr(route, "endpoint", None)
    return None


def _copy_routes_except(source: APIRouter, target: APIRouter, excluded: set[tuple[str, str]]) -> None:
    normalized = {(path, method.upper()) for path, method in excluded}
    for route in source.routes:
        path = str(getattr(route, "path", ""))
        methods = {str(value).upper() for value in (getattr(route, "methods", None) or set())}
        if any((path, method) in normalized for method in methods):
            continue
        target.routes.append(route)


def _catalog_with_legacy_compat(config: dict[str, Any]) -> dict[str, Any]:
    """Preserve today's discovery payload while adding the generic catalog."""
    payload = legacy._discover_local_models(config)
    catalog = discover_video_models(config)
    payload["catalog_items"] = catalog.get("items", [])
    payload["catalog_count"] = int(catalog.get("count", 0))
    payload["hf_cache"] = catalog.get("hf_cache")
    return payload


def get_router(plugin_manifest: dict | None = None) -> APIRouter:
    legacy_router = legacy.get_router(plugin_manifest)
    legacy_generate = _route_endpoint(legacy_router, "/engine/generate", "POST")
    if legacy_generate is None:
        raise RuntimeError("DuckMotion legacy generate route was not found.")

    router = APIRouter()
    _copy_routes_except(
        legacy_router,
        router,
        {
            ("/models/discover", "GET"),
            ("/engine/generate", "POST"),
        },
    )

    @router.get("/models/catalog")
    def model_catalog() -> dict[str, Any]:
        return discover_video_models(legacy._load_config())

    @router.get("/models/discover")
    def discover_models() -> dict[str, Any]:
        return _catalog_with_legacy_compat(legacy._load_config())

    @router.post("/engine/generate")
    def engine_generate(payload: legacy.GeneratePayload) -> dict[str, Any]:
        config = legacy._load_config()
        source = str(config.get("model_id_or_path") or "").strip()
        descriptor = describe_video_model(source)

        if not descriptor.supported:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "Model runtime unavailable.",
                    "message": (
                        f"Video model '{descriptor.name}' was discovered successfully, but its runtime "
                        "adapter is not available in this DuckMotion build yet."
                    ),
                    "model": descriptor.to_public_dict(),
                },
            )

        image_path = str(getattr(payload, "image_path", "") or "").strip()
        if descriptor.capabilities.source_image_required and not image_path:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Source image required.",
                    "message": f"Video model '{descriptor.name}' requires a source image.",
                },
            )
        if image_path and not descriptor.capabilities.image_to_video:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Source image unsupported.",
                    "message": f"Video model '{descriptor.name}' does not support image-to-video input.",
                },
            )

        return legacy_generate(payload)

    return router


# Keep WebbDuck's cross-plugin VRAM cleanup compatibility while backend.py is
# still the owner of the Wan pipeline cache.
def _unload_pipeline() -> None:
    legacy._unload_pipeline()
