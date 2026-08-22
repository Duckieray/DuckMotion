"""Model-driven DuckMotion plugin composition root."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


PLUGIN_ROOT = Path(__file__).resolve().parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from job_runtime import VideoJobCoordinator
from ltx_backend import ensure_registered as ensure_ltx_registered
from model_discovery import discover_video_models
from model_runtime import backend_resolver, describe_video_model
from runtime_services import VideoRuntimeServices
from wan_backend import ensure_registered as ensure_wan_registered


def _load_wan_implementation():
    """Load the remaining Wan-specific implementation primitives.

    backend.py no longer owns DuckMotion's public routing or job orchestration.
    It is a temporary implementation module until its reusable persistence,
    gallery and Wan pipeline pieces are physically split into smaller modules.
    """
    spec = importlib.util.spec_from_file_location(
        "duckmotion_wan_implementation",
        PLUGIN_ROOT / "backend.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load DuckMotion Wan implementation module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wan_impl = _load_wan_implementation()
services = VideoRuntimeServices(wan_impl)
job_coordinator = VideoJobCoordinator(services)


def _register_installed_backends() -> None:
    ensure_wan_registered(wan_impl)
    ensure_ltx_registered()


class GeneratePayload(BaseModel):
    image_path: str | None = None
    prompt: str
    negative_prompt: str | None = ""
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    fps: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None


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


def get_router(plugin_manifest: dict | None = None) -> APIRouter:
    implementation_router = wan_impl.get_router(plugin_manifest)
    router = APIRouter()
    _copy_routes_except(
        implementation_router,
        router,
        {
            ("/models", "GET"),
            ("/models/discover", "GET"),
            ("/engine/generate", "POST"),
        },
    )

    @router.get("/models")
    @router.get("/models/discover")
    def models() -> dict[str, Any]:
        return discover_video_models(wan_impl._load_config())

    @router.post("/engine/generate")
    def engine_generate(payload: GeneratePayload) -> dict[str, Any]:
        config = wan_impl._load_config()
        source = str(config.get("model_id_or_path") or "").strip()
        descriptor = describe_video_model(source)
        if not descriptor.supported:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "Model runtime unavailable.",
                    "message": f"No runnable backend is installed for video model '{descriptor.name}'.",
                    "model": descriptor.to_public_dict(),
                },
            )

        _register_installed_backends()
        try:
            # Resolve before queueing so a descriptor cannot claim runnable
            # status without an installed backend actually accepting it.
            backend_resolver.resolve(descriptor)
            job = job_coordinator.submit(descriptor, payload.model_dump(), config)
        except (ValueError, LookupError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {"ok": True, "job": job, "model": descriptor.to_public_dict()}

    return router


def _unload_pipeline() -> None:
    """Release in-process Wan state before another WebbDuck GPU owner runs."""
    services.unload_wan_pipeline()
