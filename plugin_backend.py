"""Model-driven DuckMotion plugin composition root."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel


PLUGIN_ROOT = Path(__file__).resolve().parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from runtime_paths import configure_default_runtime_env

# Prepared runtimes live at deterministic DuckMotion-owned paths. Advanced users
# may still override individual interpreters with DUCKMOTION_*_PYTHON, but normal
# installs should never need to export those variables manually.
configure_default_runtime_env()

from lora_runtime import LoraAwareVideoJobCoordinator, public_lora_catalog, supports_loras
from ltx_backend import ensure_registered as ensure_ltx_registered
from ltx_convrot_backend import ensure_registered as ensure_ltx_convrot_registered
from model_discovery import discover_video_models
from model_runtime import backend_resolver, describe_video_model
from runtime_services import VideoRuntimeServices
from runtime_surfaces import VideoConfigPayload, VideoRuntimeSurfaces
from storage_api import build_storage_router
from wan_backend import ensure_registered as ensure_wan_registered
from webbduck_media import recent_webbduck_images


services = VideoRuntimeServices()
job_coordinator = LoraAwareVideoJobCoordinator(services)


def _register_installed_backends() -> None:
    ensure_wan_registered()
    ensure_ltx_registered()
    ensure_ltx_convrot_registered()


def _weights_state(item: dict[str, Any]) -> dict[str, Any]:
    source = str(item.get("source") or "").strip()
    if not source:
        return {"present": False, "source": "", "kind": "unknown"}
    path = Path(source).expanduser()
    if path.exists():
        return {
            "present": True,
            "source": str(path),
            "kind": "file" if path.is_file() else "directory",
        }
    if "/" in source and not source.startswith(("/", "./", "../")):
        return {"present": None, "source": source, "kind": "remote_or_uncached"}
    return {"present": False, "source": source, "kind": "missing"}


def _effective_public_model(descriptor, runtime: dict[str, Any]) -> dict[str, Any]:
    """Apply private runtime/profile semantics to an architecture-free payload.

    Execution profile IDs remain diagnostic/internal. The browser only needs the
    resulting effective defaults and constraints for the selected checkpoint.
    """
    public = descriptor.to_public_dict()
    defaults = dict(public.get("defaults") or {})
    constraints = dict(public.get("constraints") or {})
    profile_defaults = runtime.get("profile_defaults")
    profile_constraints = runtime.get("profile_constraints")
    if isinstance(profile_defaults, dict):
        defaults.update(profile_defaults)
    if isinstance(profile_constraints, dict):
        constraints.update(profile_constraints)
    public["defaults"] = defaults
    public["constraints"] = constraints
    return public


def _resolved_catalog_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = services.load_config()
    catalog = discover_video_models(config)
    _register_installed_backends()
    rows: list[dict[str, Any]] = []
    for item in catalog.get("items") or []:
        if not isinstance(item, dict):
            continue
        identity = str(
            item.get("repo_id") or item.get("source") or item.get("name") or ""
        ).strip()
        name = str(item.get("name") or identity).strip()
        descriptor = describe_video_model(identity, name=name)
        if not descriptor.supported:
            runtime = {
                "ready": False,
                "reason": "No installed backend currently implements this model's runnable workflow.",
            }
        else:
            try:
                runtime = backend_resolver.readiness(descriptor)
            except Exception as exc:
                runtime = {"ready": False, "reason": str(exc or exc.__class__.__name__)}
        public = _effective_public_model(descriptor, runtime)
        row = {
            **public,
            "location": item.get("location"),
            "weights": _weights_state(item),
            "runtime": runtime,
            "ready": bool(public.get("supported") and runtime.get("ready")),
        }
        if item.get("repo_id"):
            row["repo_id"] = item.get("repo_id")
        rows.append(row)
    return rows, catalog


def _runtime_readiness() -> dict[str, Any]:
    rows, catalog = _resolved_catalog_rows()
    supported_rows = [row for row in rows if row.get("supported")]
    return {
        "ready": bool(supported_rows) and all(row.get("ready") for row in supported_rows),
        "count": len(rows),
        "items": rows,
        "hf_cache": catalog.get("hf_cache"),
    }


def _public_model_catalog() -> dict[str, Any]:
    """Return effective public models without backend/profile routing metadata."""
    rows, catalog = _resolved_catalog_rows()
    items: list[dict[str, Any]] = []
    for row in rows:
        item = {
            key: value
            for key, value in row.items()
            if key not in {"runtime", "weights"}
        }
        items.append(item)
    return {
        "count": len(items),
        "items": items,
        "hf_cache": catalog.get("hf_cache"),
    }


runtime_surfaces = VideoRuntimeSurfaces(
    services=services,
    resolver=backend_resolver,
    describe_model=describe_video_model,
    discover_models=discover_video_models,
    register_backends=_register_installed_backends,
)


class LoraSelectionPayload(BaseModel):
    name: str
    weight: float = 1.0


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
    loras: list[LoraSelectionPayload] | None = None
    # Optional generic I2V quality hint. Backends that advertise support through
    # public constraints may interpret model/identity/locked; others can ignore it.
    i2v_stability: str | None = None


def get_router(plugin_manifest: dict | None = None) -> APIRouter:
    del plugin_manifest
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, Any]:
        return runtime_surfaces.health()

    @router.get("/runtime-readiness")
    def runtime_readiness() -> dict[str, Any]:
        """Probe every discovered model/runtime without loading model weights."""
        return _runtime_readiness()

    @router.get("/config")
    def get_config() -> dict[str, Any]:
        return runtime_surfaces.get_config()

    @router.post("/config")
    def set_config(payload: VideoConfigPayload) -> dict[str, Any]:
        return runtime_surfaces.set_config(payload)

    @router.get("/models")
    @router.get("/models/discover")
    def models() -> dict[str, Any]:
        # Discovery remains pure; this composition layer overlays privately
        # resolved recipe defaults/constraints before returning public models.
        return _public_model_catalog()

    @router.get("/loras")
    def loras() -> dict[str, Any]:
        """List LoRAs applicable to the currently selected video model."""
        config = services.load_config()
        source = str(config.get("model_id_or_path") or "").strip()
        descriptor = describe_video_model(source) if source else None
        return public_lora_catalog(supported=supports_loras(descriptor))

    @router.get("/engine/runtime")
    def engine_runtime() -> dict[str, Any]:
        return services.runtime_profile_safe()

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

    @router.get("/webbduck/recent-images")
    def recent_images(limit: int = Query(default=24, ge=1, le=200)) -> dict[str, Any]:
        return {"items": recent_webbduck_images(int(limit))}

    router.include_router(build_storage_router(services.storage, services.load_config))
    return router


def _unload_pipeline() -> None:
    """Release all installed DuckMotion runtime resources."""
    runtime_surfaces.unload()
