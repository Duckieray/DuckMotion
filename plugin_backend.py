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
from runtime_surfaces import VideoConfigPayload, VideoRuntimeSurfaces
from storage_api import build_storage_router
from wan_backend import ensure_registered as ensure_wan_registered


def _load_wan_implementation():
    """Load the remaining Wan pipeline implementation from backend.py.

    Public routing, orchestration, health/config/status, persistence, staging,
    and gallery ownership have moved to architecture-neutral modules. The old
    module is now an implementation source for the mature Wan pipeline plus a
    few support routes pending the final UI/server cutover.
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


runtime_surfaces = VideoRuntimeSurfaces(
    services=services,
    resolver=backend_resolver,
    describe_model=describe_video_model,
    discover_models=discover_video_models,
    register_backends=_register_installed_backends,
)


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
            ("/health", "GET"),
            ("/config", "GET"),
            ("/config", "POST"),
            ("/models", "GET"),
            ("/models/discover", "GET"),
            ("/engine/status", "GET"),
            ("/engine/unload", "POST"),
            ("/engine/generate", "POST"),
            ("/engine/jobs", "GET"),
            ("/engine/jobs/{job_id}", "GET"),
            ("/engine/cancel", "POST"),
            ("/jobs/clear", "POST"),
            ("/staging/upload", "POST"),
            ("/staging/from-webbduck", "POST"),
            ("/staging", "GET"),
            ("/staging/{name}", "DELETE"),
            ("/gallery", "GET"),
            ("/gallery/file/{run_id}/{filename}", "GET"),
        },
    )

    @router.get("/health")
    def health() -> dict[str, Any]:
        return runtime_surfaces.health()

    @router.get("/config")
    def get_config() -> dict[str, Any]:
        return runtime_surfaces.get_config()

    @router.post("/config")
    def set_config(payload: VideoConfigPayload) -> dict[str, Any]:
        return runtime_surfaces.set_config(payload)

    @router.get("/models")
    @router.get("/models/discover")
    def models() -> dict[str, Any]:
        return discover_video_models(services.load_config())

    @router.get("/engine/status")
    def engine_status() -> dict[str, Any]:
        return runtime_surfaces.engine_status()

    @router.post("/engine/unload")
    def engine_unload() -> dict[str, Any]:
        return runtime_surfaces.unload()

    @router.post("/engine/generate")
    def engine_generate(payload: GeneratePayload) -> dict[str, Any]:
        config = services.load_config()
        source = str(config.get("model_id_or_path") or "").strip()
        if not source:
            raise HTTPException(status_code=422, detail="Select a video model before generation.")
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
            backend_resolver.resolve(descriptor)
            job = job_coordinator.submit(descriptor, payload.model_dump(), config)
        except (ValueError, LookupError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {"ok": True, "job": job, "model": descriptor.to_public_dict()}

    router.include_router(build_storage_router(services.storage, services.load_config))
    return router


def _unload_pipeline() -> None:
    """Release all installed DuckMotion runtime resources."""
    runtime_surfaces.unload()
