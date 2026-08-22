"""Model-driven DuckMotion plugin composition root.

The public contract is checkpoint/model driven. backend.py is retained only as
the current Wan implementation and for shared staging/gallery/job persistence
while those pieces are progressively split into generic modules.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import uuid
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


PLUGIN_ROOT = Path(__file__).resolve().parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from ltx_backend import ensure_registered as ensure_ltx_registered
from model_discovery import discover_video_models
from model_runtime import backend_resolver, describe_video_model


def _load_wan_backend():
    spec = importlib.util.spec_from_file_location(
        "duckmotion_wan_backend",
        PLUGIN_ROOT / "backend.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load DuckMotion Wan backend module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wan = _load_wan_backend()


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


def _snap_ltx_dimension(value: int) -> int:
    return max(256, (int(value) // 32) * 32)


def _snap_ltx_frames(value: int) -> int:
    value = max(9, int(value))
    return max(9, ((value - 1) // 8) * 8 + 1)


def _ltx_params(payload: GeneratePayload, descriptor) -> dict[str, Any]:
    defaults = descriptor.defaults or {}
    return {
        "prompt": str(payload.prompt or "").strip(),
        "width": _snap_ltx_dimension(int(payload.width or defaults.get("width") or 768)),
        "height": _snap_ltx_dimension(int(payload.height or defaults.get("height") or 512)),
        "num_frames": _snap_ltx_frames(int(payload.num_frames or defaults.get("num_frames") or 121)),
        "fps": int(payload.fps or defaults.get("fps") or 24),
        "guidance_scale": 1.0,
        "seed": int(payload.seed) if payload.seed is not None else None,
    }


def _submit_ltx_job(payload: GeneratePayload, config: dict[str, Any], descriptor) -> dict[str, Any]:
    prompt = str(payload.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt is required.")

    src_path = None
    raw_image = str(payload.image_path or "").strip()
    if raw_image:
        src_path = wan._resolve_input_image(raw_image)

    now = wan._now()
    job_id = f"dm_{uuid.uuid4().hex[:12]}"
    input_payload = {
        "image_path": str(src_path) if src_path is not None else "",
        "image_name": src_path.name if src_path is not None else "",
        "web_path": wan._safe_web_path_for_any(src_path) if src_path is not None else None,
    }
    row = {
        "job_id": job_id,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "cancel_requested": False,
        "progress": {"stage": "queued", "percent": 0},
        "model": descriptor.to_public_dict(),
        "config_snapshot": {
            "model_id_or_path": descriptor.source,
            "output_dir": str(wan._resolve_output_dir(config)),
        },
        "input": input_payload,
        "params": _ltx_params(payload, descriptor),
        "error": None,
        "warnings": [],
        "outputs": [],
    }
    wan._upsert_job(row)
    threading.Thread(
        target=_run_ltx_job,
        args=(job_id, config, descriptor),
        name=f"duckmotion-ltx-{job_id}",
        daemon=True,
    ).start()
    return row


def _run_ltx_job(job_id: str, config: dict[str, Any], descriptor) -> None:
    row = wan._get_job(job_id)
    if row is None or row.get("status") == "canceled":
        return

    lease_token = None
    try:
        row["status"] = "running"
        row["started_at"] = wan._now()
        row["updated_at"] = wan._now()
        row["progress"] = {"stage": "waiting_for_gpu", "percent": 5}
        wan._upsert_job(row)

        lease_attempt = wan.acquire_gpu_lease_blocking(
            owner="duckmotion",
            owner_kind="plugin",
            label="ltx25",
            job_id=job_id,
        )
        lease = lease_attempt.get("lease") if isinstance(lease_attempt, dict) else None
        if isinstance(lease, dict):
            lease_token = str(lease.get("token") or "").strip() or None

        wan._unload_pipeline()
        row = wan._get_job(job_id) or row
        if row.get("cancel_requested"):
            raise RuntimeError("LTX generation cancelled")
        row["progress"] = {"stage": "loading_pipeline", "percent": 15}
        row["updated_at"] = wan._now()
        wan._upsert_job(row)

        ensure_ltx_registered()
        runtime = backend_resolver.resolve(descriptor)
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{job_id}"
        run_dir = wan._resolve_output_dir(config) / run_id
        request = dict(row.get("params") or {})
        request["image_path"] = str((row.get("input") or {}).get("image_path") or "")

        def is_cancelled() -> bool:
            current = wan._get_job(job_id)
            return bool(current and current.get("cancel_requested"))

        result = runtime.generate(
            descriptor,
            request,
            output_dir=run_dir,
            is_cancelled=is_cancelled,
        )

        meta_path = run_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        except Exception:
            meta = {}
        meta.update(
            {
                "plugin": "duckmotion",
                "job_id": job_id,
                "run_id": run_id,
                "created_at": wan._now(),
                "input": row.get("input") or {},
                "params": row.get("params") or {},
                "model": descriptor.to_public_dict(),
                "frame_count": int(result.get("frame_count") or (row.get("params") or {}).get("num_frames") or 0),
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        output_info = wan._gallery_item_from_run(run_dir)
        if output_info is None:
            raise RuntimeError("LTX generation completed but no gallery artifact could be built")

        row = wan._get_job(job_id) or row
        row["status"] = "completed"
        row["error"] = None
        row["outputs"] = [output_info]
        row["updated_at"] = wan._now()
        row["finished_at"] = wan._now()
        row["progress"] = {"stage": "completed", "percent": 100}
        wan._upsert_job(row)
    except Exception as exc:
        row = wan._get_job(job_id) or row
        canceled = bool(row.get("cancel_requested")) or "cancel" in str(exc).lower()
        row["status"] = "canceled" if canceled else "failed"
        row["error"] = None if canceled else str(exc or exc.__class__.__name__)
        row["updated_at"] = wan._now()
        row["finished_at"] = wan._now()
        row["progress"] = {"stage": row["status"], "percent": row.get("progress", {}).get("percent", 0)}
        wan._upsert_job(row)
    finally:
        if lease_token:
            try:
                wan.release_gpu_lease(token=lease_token)
            except Exception:
                pass


def get_router(plugin_manifest: dict | None = None) -> APIRouter:
    wan_router = wan.get_router(plugin_manifest)
    wan_generate = _route_endpoint(wan_router, "/engine/generate", "POST")
    if wan_generate is None:
        raise RuntimeError("DuckMotion Wan generate route was not found.")

    router = APIRouter()
    _copy_routes_except(
        wan_router,
        router,
        {
            ("/models/discover", "GET"),
            ("/engine/generate", "POST"),
        },
    )

    @router.get("/models")
    @router.get("/models/discover")
    def models() -> dict[str, Any]:
        return discover_video_models(wan._load_config())

    @router.post("/engine/generate")
    def engine_generate(payload: GeneratePayload) -> dict[str, Any]:
        config = wan._load_config()
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

        image_path = str(payload.image_path or "").strip()
        if descriptor.capabilities.source_image_required and not image_path:
            raise HTTPException(status_code=422, detail="The selected model requires a source image.")
        if image_path and not descriptor.capabilities.image_to_video:
            raise HTTPException(status_code=422, detail="The selected model does not support image conditioning.")

        if descriptor.backend == "ltx25_isolated":
            job = _submit_ltx_job(payload, config, descriptor)
            return {"ok": True, "job": job, "model": descriptor.to_public_dict()}

        # Current Wan implementation remains internally reusable until it is
        # moved into its own VideoBackend. This is not a public compatibility
        # contract and can be deleted once the Wan adapter migration lands.
        if not image_path:
            raise HTTPException(status_code=422, detail="The selected video model requires an image.")
        wan_payload = wan.GeneratePayload(**payload.model_dump())
        return wan_generate(wan_payload)

    return router


def _unload_pipeline() -> None:
    """Release currently loaded in-process Wan state before another GPU owner runs."""
    wan._unload_pipeline()
