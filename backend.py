"""DuckMotion (Wan2.2) web plugin backend for WebbDuck.

Local diffusers-based runtime scaffold:
- WebbDuck-compatible runtime profile (device/dtype) reuse
- plugin-local job queue + worker
- image staging + WebbDuck image copy
- configurable output directory with plugin gallery
- Wan2.2 image-to-video generation via diffusers (lazy imports)

Notes:
- Requires a diffusers build that includes ``WanImageToVideoPipeline``.
- Cancellation is best-effort for queued jobs; running diffusers jobs cannot always be interrupted safely.
"""

from __future__ import annotations

import json
import inspect
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import gc
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from webbduck.server.storage import BASE, resolve_web_path, to_web_path
    from webbduck.core.gpu_lease import acquire_gpu_lease_blocking, get_gpu_lease, release_gpu_lease
except ModuleNotFoundError:
    # Child-process invocation executes this file directly; ensure repo root is importable.
    _parents = Path(__file__).resolve().parents
    _repo_root = _parents[3] if len(_parents) > 3 else Path.cwd().resolve()
    _candidates = [_repo_root.parent, _repo_root]
    for _candidate in _candidates:
        _value = str(_candidate)
        if _value not in sys.path:
            sys.path.insert(0, _value)
    from webbduck.server.storage import BASE, resolve_web_path, to_web_path
    from webbduck.core.gpu_lease import acquire_gpu_lease_blocking, get_gpu_lease, release_gpu_lease

PLUGIN_ROOT = Path(__file__).resolve().parent
STATE_DIR = Path.home() / ".webbduck" / "plugin_state"
CONFIG_FILE = STATE_DIR / "duckmotion_config.json"
JOBS_FILE = STATE_DIR / "duckmotion_jobs.json"
STAGING_DIR = BASE / "duckmotion_staging"

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
DEFAULT_OUTPUT_DIR = BASE / "duckmotion_videos"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv"}

_LOCK = threading.RLock()
_JOB_QUEUE: queue.Queue[str] = queue.Queue()
_WORKER_THREAD: threading.Thread | None = None
_PIPELINE_LOCK = threading.RLock()
_PIPELINE: Any | None = None
_PIPELINE_KEY: tuple[str, str, str] | None = None
_PIPELINE_MEMORY_STRATEGY: dict[str, Any] | None = None
_LOG = logging.getLogger("duckmotion.backend")


def _webbduck_path_hints() -> dict[str, Path | None]:
    """Resolve WebbDuck-owned path hints when available.

    Keeps DuckMotion aligned with the same roots WebbDuck uses internally.
    """
    hints: dict[str, Path | None] = {
        "root": None,
        "models_root": None,
        "checkpoint_root": None,
        "checkpoint_wan": None,
        "hf_cache": None,
    }
    hints["checkpoint_wan_candidates"] = []  # type: ignore[index]
    try:
        from webbduck.models import registry as model_registry  # lazy import

        root = Path(getattr(model_registry, "ROOT", "")).resolve() if getattr(model_registry, "ROOT", None) else None
        models_root = (
            Path(getattr(model_registry, "MODELS_ROOT", "")).expanduser().resolve()
            if getattr(model_registry, "MODELS_ROOT", None)
            else None
        )
        checkpoint_root = (
            Path(getattr(model_registry, "CHECKPOINT_ROOT", "")).resolve()
            if getattr(model_registry, "CHECKPOINT_ROOT", None)
            else None
        )
        hf_cache = (
            Path(getattr(model_registry, "HF_CACHE", "")).expanduser().resolve()
            if getattr(model_registry, "HF_CACHE", None)
            else None
        )

        # If the registry resolved models_root to the repo root (i.e. WEBBDUCK_MODELS_DIR
        # was not set at import time), read it directly from the env var as a guard against
        # import-order issues.  This keeps DuckMotion aligned with however WebbDuck was launched.
        env_models_dir = _normalize_path_string(os.getenv("WEBBDUCK_MODELS_DIR") or "")
        if env_models_dir and (models_root is None or models_root == root):
            models_root = Path(env_models_dir).expanduser().resolve()

        hints["root"] = root
        hints["models_root"] = models_root
        hints["checkpoint_root"] = checkpoint_root
        wan_candidates = _checkpoint_wan_candidates(checkpoint_root=checkpoint_root, models_root=models_root, root=root)
        hints["checkpoint_wan_candidates"] = wan_candidates  # type: ignore[index]
        hints["checkpoint_wan"] = wan_candidates[0] if wan_candidates else None
        hints["hf_cache"] = hf_cache
    except Exception:
        # Fallback to cwd-based assumptions if WebbDuck internals are unavailable.
        root = Path.cwd().resolve()
        models_root_raw = _normalize_path_string(os.getenv("WEBBDUCK_MODELS_DIR") or "")
        models_root = Path(models_root_raw).expanduser().resolve() if models_root_raw else root
        hints["root"] = root
        hints["models_root"] = models_root
        checkpoint_root = _first_existing_path(
            [
                models_root / "checkpoints" / "sdxl",
                models_root / "checkpoint" / "sdxl",
                models_root / "checkpoints",
                models_root / "checkpoint",
            ]
        ).resolve()
        hints["checkpoint_root"] = checkpoint_root
        wan_candidates = _checkpoint_wan_candidates(checkpoint_root=checkpoint_root, models_root=models_root, root=root)
        hints["checkpoint_wan_candidates"] = wan_candidates  # type: ignore[index]
        hints["checkpoint_wan"] = wan_candidates[0] if wan_candidates else None
        hints["hf_cache"] = (Path.home() / ".cache" / "huggingface" / "hub").resolve()
    return hints





def _default_model_source() -> str:
    env_model = _normalize_str(
        os.getenv("DUCKMOTION_MODEL_ID_OR_PATH") or os.getenv("DUCKMOTION_MODEL_ID") or ""
    )
    if env_model:
        return env_model

    hints = _webbduck_path_hints()
    for checkpoint_wan in _wan_search_roots_from_hints(hints):
        if not checkpoint_wan.exists():
            continue
        discovered = _scan_local_dirs_for_wan_diffusers(checkpoint_wan)
        if discovered:
            # Prefer latest modified local model for one-click startup.
            discovered.sort(key=lambda p: _safe_mtime(p), reverse=True)
            return str(discovered[0])

    return DEFAULT_MODEL_ID


def _now() -> float:
    return time.time()


def _write_progress_file(progress_path: Path | None, stage: str, percent: int, detail: str = "") -> None:
    if progress_path is None:
        return
    try:
        payload = {
            "stage": str(stage or ""),
            "percent": int(percent),
            "detail": str(detail or ""),
            "updated_at": _now(),
        }
        progress_path.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


class ConfigPayload(BaseModel):
    model_id_or_path: str | None = None
    models_dir: str | None = None
    output_dir: str | None = None
    runtime_backend: str | None = None
    gguf_transformer_path: str | None = None
    memory_policy: str | None = None
    default_width: int | None = None
    default_height: int | None = None
    default_frames: int | None = None
    default_fps: int | None = None
    default_steps: int | None = None
    default_guidance_scale: float | None = None


class GeneratePayload(BaseModel):
    image_path: str
    prompt: str
    negative_prompt: str | None = ""
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    fps: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None


class CancelPayload(BaseModel):
    job_id: str


def get_router(_plugin_manifest: dict | None = None) -> APIRouter:
    _ensure_worker_started()
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, Any]:
        config = _load_config()
        install = _detect_installation(config)
        runtime = _get_runtime_profile_safe()
        runtime_profile = runtime.get("profile") if isinstance(runtime, dict) else None
        compatibility = (
            _get_current_setup_compatibility(config, runtime_profile)
            if isinstance(runtime_profile, dict)
            else {"compatible": False, "reason": runtime.get("error") or "Runtime profile unavailable."}
        )
        if isinstance(install, dict):
            install["compatibility"] = compatibility
        engine = _engine_status(config, runtime)
        ready = {
            "plugin_enabled": bool(install.get("model_source_configured")),
            "generation_prereqs": bool(
                install.get("diffusers_ready") and install.get("output_dir_writable") and compatibility.get("compatible")
            ),
            "runtime_ready": bool(runtime.get("ok")),
            "engine_ready": bool(
                install.get("diffusers_ready")
                and install.get("model_source_configured")
                and install.get("output_dir_writable")
                and runtime.get("ok")
                and compatibility.get("compatible")
            ),
        }
        return {
            "ok": True,
            "phase": 2,
            "mode": "local-diffusers-runtime",
            "config": _public_config(config),
            "installation": install,
            "runtime": runtime,
            "engine": engine,
            "ready": ready,
            "notes": [
                "DuckMotion runs as a local WebbDuck plugin runtime.",
                "Device/dtype resolution reuses WebbDuck runtime profile behavior.",
            ],
        }

    @router.get("/config")
    def get_config() -> dict[str, Any]:
        return {"config": _public_config(_load_config())}

    @router.get("/models/discover")
    def discover_models() -> dict[str, Any]:
        config = _load_config()
        return _discover_local_models(config)

    @router.post("/config")
    def set_config(payload: ConfigPayload) -> dict[str, Any]:
        current = _load_config()
        next_config = dict(current)
        if payload.model_id_or_path is not None:
            model_val = _normalize_str(payload.model_id_or_path)
            if model_val.startswith("gguf:"):
                model_val = model_val[5:]
            next_config["model_id_or_path"] = model_val
        if payload.models_dir is not None:
            next_config["models_dir"] = _normalize_path_string(payload.models_dir)
        if payload.output_dir is not None:
            next_config["output_dir"] = _normalize_path_string(payload.output_dir)
        if payload.runtime_backend is not None:
            next_config["runtime_backend"] = _normalize_runtime_backend(payload.runtime_backend)
        if payload.gguf_transformer_path is not None:
            next_config["gguf_transformer_path"] = _normalize_path_string(payload.gguf_transformer_path)
        if payload.memory_policy is not None:
            next_config["memory_policy"] = _normalize_memory_policy(payload.memory_policy)

        next_config["default_width"] = _clamp_int(
            payload.default_width if payload.default_width is not None else current.get("default_width"),
            default=832,
            lo=256,
            hi=1920,
        )
        next_config["default_height"] = _clamp_int(
            payload.default_height if payload.default_height is not None else current.get("default_height"),
            default=480,
            lo=256,
            hi=1920,
        )
        next_config["default_frames"] = _clamp_int(
            payload.default_frames if payload.default_frames is not None else current.get("default_frames"),
            default=81,
            lo=8,
            hi=241,
        )
        next_config["default_fps"] = _clamp_int(
            payload.default_fps if payload.default_fps is not None else current.get("default_fps"),
            default=16,
            lo=1,
            hi=60,
        )
        next_config["default_steps"] = _clamp_int(
            payload.default_steps if payload.default_steps is not None else current.get("default_steps"),
            default=30,
            lo=1,
            hi=120,
        )
        next_config["default_guidance_scale"] = _clamp_float(
            payload.default_guidance_scale
            if payload.default_guidance_scale is not None
            else current.get("default_guidance_scale"),
            default=5.0,
            lo=0.0,
            hi=20.0,
        )

        _save_config(next_config)
        return {"ok": True, "config": _public_config(next_config)}

    @router.get("/engine/runtime")
    def engine_runtime() -> dict[str, Any]:
        return _get_runtime_profile_safe()

    @router.get("/engine/status")
    def engine_status() -> dict[str, Any]:
        config = _load_config()
        return _engine_status(config, _get_runtime_profile_safe())

    @router.post("/engine/unload")
    def engine_unload() -> dict[str, Any]:
        _unload_pipeline()
        return {"ok": True, "message": "DuckMotion diffusers pipeline cache cleared."}

    @router.post("/engine/generate")
    def engine_generate(payload: GeneratePayload) -> dict[str, Any]:
        config = _load_config()
        runtime_profile = _get_runtime_profile_or_raise()
        compatibility = _get_current_setup_compatibility(config, runtime_profile)
        if not compatibility.get("compatible"):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "Incompatible with current setup.",
                    "message": str(compatibility.get("reason") or "DuckMotion backend is incompatible with current setup."),
                    "compatibility": compatibility,
                },
            )
        params = _normalize_generate_params(payload, config)
        safety = _evaluate_job_safety(params, config, runtime_profile)
        if not safety.get("ok"):
            raise HTTPException(status_code=409, detail=safety)
        job = _submit_job(payload=payload, config=config)
        return {"ok": True, "job": job, "safety": safety}

    @router.get("/engine/jobs")
    def engine_jobs(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        rows = _list_jobs(limit=int(limit))
        return {"jobs": rows}

    @router.get("/engine/jobs/{job_id}")
    def engine_job(job_id: str) -> dict[str, Any]:
        row = _get_job(job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return {"job": row}

    @router.post("/engine/cancel")
    def engine_cancel(payload: CancelPayload) -> dict[str, Any]:
        result = _cancel_job(payload.job_id)
        if not result["ok"]:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    @router.post("/jobs/clear")
    def clear_jobs() -> dict[str, Any]:
        with _LOCK:
            _save_jobs_locked([])
        return {"ok": True}

    @router.post("/staging/upload")
    async def staging_upload(image: UploadFile = File(...)) -> dict[str, Any]:
        filename = str(image.filename or "").strip()
        if not filename:
            raise HTTPException(status_code=400, detail="Missing filename.")
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_IMAGE_SUFFIXES:
            raise HTTPException(status_code=400, detail="Unsupported image type.")

        STAGING_DIR.mkdir(exist_ok=True, parents=True)
        stem = _safe_stem(Path(filename).stem) or "input"
        out_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{stem}{suffix}"
        out_path = STAGING_DIR / out_name

        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty upload.")
        out_path.write_bytes(data)
        return {"ok": True, "item": _staging_item(out_path)}

    @router.post("/staging/from-webbduck")
    def staging_from_webbduck(path: str = Form(...)) -> dict[str, Any]:
        src = resolve_web_path(path)
        if not src.exists() or not src.is_file():
            raise HTTPException(status_code=404, detail="Source image not found.")
        if src.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise HTTPException(status_code=400, detail="Source file is not a supported image.")

        STAGING_DIR.mkdir(exist_ok=True, parents=True)
        stem = _safe_stem(src.stem) or "webbduck"
        out_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{stem}{src.suffix.lower()}"
        out_path = STAGING_DIR / out_name
        out_path.write_bytes(src.read_bytes())
        return {"ok": True, "item": _staging_item(out_path)}

    @router.get("/staging")
    def list_staging(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        STAGING_DIR.mkdir(exist_ok=True, parents=True)
        rows = [p for p in STAGING_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
        rows.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return {"items": [_staging_item(p) for p in rows[: int(limit)]]}

    @router.delete("/staging/{name}")
    def delete_staging(name: str) -> dict[str, Any]:
        safe_name = Path(str(name or "")).name
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid file name.")
        path = STAGING_DIR / safe_name
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="Staged file not found.")
        path.unlink()
        return {"ok": True, "deleted": safe_name}

    @router.get("/webbduck/recent-images")
    def recent_webbduck_images(limit: int = Query(default=24, ge=1, le=200)) -> dict[str, Any]:
        return {"items": _recent_webbduck_images(limit=int(limit))}

    @router.get("/gallery")
    def gallery(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        config = _load_config()
        items = _scan_gallery(_resolve_output_dir(config), limit=int(limit))
        return {"items": items}

    @router.get("/gallery/file/{run_id}/{filename}")
    def gallery_file(run_id: str, filename: str) -> FileResponse:
        config = _load_config()
        root = _resolve_output_dir(config)
        safe_run = _safe_stem(run_id)
        safe_name = Path(filename).name
        if not safe_run or not safe_name:
            raise HTTPException(status_code=400, detail="Invalid path.")
        target = (root / safe_run / safe_name).resolve()
        try:
            target.relative_to(root.resolve())
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid path traversal.") from exc
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found.")
        return FileResponse(target)

    return router


# ---------------------------------------------------------------------------
# Config + readiness
# ---------------------------------------------------------------------------


def _default_config() -> dict[str, Any]:
    hints = _webbduck_path_hints()
    default_models_dir = _normalize_path_string(os.getenv("DUCKMOTION_MODELS_DIR") or "")
    if not default_models_dir and isinstance(hints.get("hf_cache"), Path):
        default_models_dir = str(hints["hf_cache"])
    return {
        "model_id_or_path": _default_model_source(),
        "models_dir": default_models_dir,
        "output_dir": _normalize_path_string(os.getenv("DUCKMOTION_OUTPUT_DIR") or ""),
        "runtime_backend": _normalize_runtime_backend(os.getenv("DUCKMOTION_RUNTIME_BACKEND") or "auto"),
        "gguf_transformer_path": _normalize_path_string(os.getenv("DUCKMOTION_GGUF_TRANSFORMER_PATH") or ""),
        "memory_policy": _normalize_memory_policy(os.getenv("DUCKMOTION_MEMORY_POLICY") or "auto"),
        "default_width": _clamp_int(os.getenv("DUCKMOTION_DEFAULT_WIDTH"), default=832, lo=256, hi=1920),
        "default_height": _clamp_int(os.getenv("DUCKMOTION_DEFAULT_HEIGHT"), default=480, lo=256, hi=1920),
        "default_frames": _clamp_int(os.getenv("DUCKMOTION_DEFAULT_FRAMES"), default=81, lo=8, hi=241),
        "default_fps": _clamp_int(os.getenv("DUCKMOTION_DEFAULT_FPS"), default=16, lo=1, hi=60),
        "default_steps": _clamp_int(os.getenv("DUCKMOTION_DEFAULT_STEPS"), default=30, lo=1, hi=120),
        "default_guidance_scale": _clamp_float(
            os.getenv("DUCKMOTION_DEFAULT_GUIDANCE_SCALE"), default=5.0, lo=0.0, hi=20.0
        ),
    }


def _load_config() -> dict[str, Any]:
    config = _default_config()
    if not CONFIG_FILE.exists():
        return config
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return config
    if not isinstance(raw, dict):
        return config

    config["model_id_or_path"] = _normalize_str(raw.get("model_id_or_path") or config["model_id_or_path"])
    config["models_dir"] = _normalize_path_string(raw.get("models_dir") or config["models_dir"])
    config["output_dir"] = _normalize_path_string(raw.get("output_dir") or config["output_dir"])
    config["runtime_backend"] = _normalize_runtime_backend(raw.get("runtime_backend") or config["runtime_backend"])
    config["gguf_transformer_path"] = _normalize_path_string(raw.get("gguf_transformer_path") or config["gguf_transformer_path"])
    config["memory_policy"] = _normalize_memory_policy(raw.get("memory_policy") or config["memory_policy"])
    config["default_width"] = _clamp_int(raw.get("default_width"), default=config["default_width"], lo=256, hi=1920)
    config["default_height"] = _clamp_int(raw.get("default_height"), default=config["default_height"], lo=256, hi=1920)
    config["default_frames"] = _clamp_int(raw.get("default_frames"), default=config["default_frames"], lo=8, hi=241)
    config["default_fps"] = _clamp_int(raw.get("default_fps"), default=config["default_fps"], lo=1, hi=60)
    config["default_steps"] = _clamp_int(raw.get("default_steps"), default=config["default_steps"], lo=1, hi=120)
    config["default_guidance_scale"] = _clamp_float(
        raw.get("default_guidance_scale"),
        default=config["default_guidance_scale"],
        lo=0.0,
        hi=20.0,
    )
    return config


def _save_config(config: dict[str, Any]) -> None:
    STATE_DIR.mkdir(exist_ok=True, parents=True)
    payload = _public_config(config)
    CONFIG_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id_or_path": _normalize_str(config.get("model_id_or_path") or DEFAULT_MODEL_ID),
        "models_dir": _normalize_path_string(config.get("models_dir") or ""),
        "output_dir": _normalize_path_string(config.get("output_dir") or ""),
        "runtime_backend": _normalize_runtime_backend(config.get("runtime_backend") or "auto"),
        "gguf_transformer_path": _normalize_path_string(config.get("gguf_transformer_path") or ""),
        "memory_policy": _normalize_memory_policy(config.get("memory_policy") or "auto"),
        "default_width": _clamp_int(config.get("default_width"), default=832, lo=256, hi=1920),
        "default_height": _clamp_int(config.get("default_height"), default=480, lo=256, hi=1920),
        "default_frames": _clamp_int(config.get("default_frames"), default=81, lo=8, hi=241),
        "default_fps": _clamp_int(config.get("default_fps"), default=16, lo=1, hi=60),
        "default_steps": _clamp_int(config.get("default_steps"), default=30, lo=1, hi=120),
        "default_guidance_scale": _clamp_float(config.get("default_guidance_scale"), default=5.0, lo=0.0, hi=20.0),
        "resolved_output_dir": str(_resolve_output_dir(config)),
    }


def _resolve_output_dir(config: dict[str, Any]) -> Path:
    raw = _normalize_path_string(config.get("output_dir") or "")
    path = Path(raw).expanduser() if raw else DEFAULT_OUTPUT_DIR
    path.mkdir(exist_ok=True, parents=True)
    return path


def _looks_like_path(value: str) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    if raw.startswith(("./", "../", "/", "~")):
        return True
    return bool(re.match(r"^[A-Za-z]:[\\/]", raw))


def _discover_local_models(config: dict[str, Any]) -> dict[str, Any]:
    roots = _candidate_model_search_roots(config)
    items: list[dict[str, Any]] = []
    gguf_items: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_gguf_paths: set[str] = set()
    scan_errors: list[str] = []

    for root_info in roots:
        root = root_info["path"]
        source = root_info["source"]
        if source == "hf_cache":
            try:
                for row in _scan_hf_cache_for_wan_diffusers(root):
                    norm = str(Path(row["path"]).resolve())
                    if norm in seen_paths:
                        continue
                    seen_paths.add(norm)
                    items.append(row)
            except Exception as exc:
                scan_errors.append(f"{root}: {exc}")
            continue

        if not root.exists() or not root.is_dir():
            continue
        try:
            for candidate in _scan_local_dirs_for_wan_diffusers(root):
                norm = str(candidate.resolve())
                if norm in seen_paths:
                    continue
                seen_paths.add(norm)
                repo_hint = _repo_id_hint_from_local_name(candidate.name)
                items.append(
                    _discovered_model_item(
                        path=candidate,
                        source=source,
                        repo_id=repo_hint,
                        label=(f"{candidate.name} ({source})" if source != "checkpoint_wan" else candidate.name),
                    )
                )
            for gguf_path in _scan_local_files_for_wan_gguf(root):
                norm = str(gguf_path.resolve())
                if norm in seen_gguf_paths:
                    continue
                seen_gguf_paths.add(norm)
                gguf_items.append(_discovered_gguf_item(path=gguf_path, source=source))
        except Exception as exc:
            scan_errors.append(f"{root}: {exc}")

    items.sort(key=_discovered_model_sort_key)
    gguf_items.sort(key=_discovered_model_sort_key)
    gguf_candidates = _group_gguf_candidates(gguf_items)
    return {
        "items": items[:200],
        "gguf_items": gguf_items[:200],
        "gguf_candidates": gguf_candidates[:200],
        "scan_roots": [
            {
                "source": r["source"],
                "path": str(r["path"]),
                "exists": bool(r["path"].exists()),
            }
            for r in roots
        ],
        "scan_errors": scan_errors[:30],
        "default_model_id": DEFAULT_MODEL_ID,
        "runtime_plan": _plan_runtime_backend(config, None, discovered_gguf_items=gguf_items, gguf_candidates=gguf_candidates),
    }


def _first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _checkpoint_wan_candidates(*, checkpoint_root: Path | None, models_root: Path | None, root: Path | None) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()

    def add(path: Path | None) -> None:
        if path is None:
            return
        value = path.expanduser().resolve()
        key = str(value)
        if key in seen:
            return
        seen.add(key)
        out.append(value)

    if checkpoint_root is not None:
        cr = checkpoint_root.expanduser().resolve()
        name = cr.name.lower()
        if name == "sdxl":
            cp_parent = cr.parent
            add(cp_parent / "wan")
            if cp_parent.name.lower() == "checkpoint":
                add(cp_parent.parent / "checkpoints" / "wan")
            if cp_parent.name.lower() == "checkpoints":
                add(cp_parent.parent / "checkpoint" / "wan")
        elif name in {"checkpoint", "checkpoints"}:
            add(cr / "wan")
        else:
            add(cr / "wan")

    if models_root is not None:
        mr = models_root.expanduser().resolve()
        add(mr / "checkpoint" / "wan")
        add(mr / "checkpoints" / "wan")

    if root is not None:
        rr = root.expanduser().resolve()
        add(rr / "checkpoint" / "wan")
        add(rr / "checkpoints" / "wan")

    return out


def _wan_search_roots_from_hints(hints: dict[str, Any]) -> list[Path]:
    rows: list[Path] = []
    seen: set[str] = set()

    def add(path: Path | None) -> None:
        if path is None:
            return
        value = path.expanduser().resolve()
        key = str(value)
        if key in seen:
            return
        seen.add(key)
        rows.append(value)

    candidates = hints.get("checkpoint_wan_candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, Path):
                add(candidate)

    checkpoint_wan = hints.get("checkpoint_wan")
    if isinstance(checkpoint_wan, Path):
        add(checkpoint_wan)

    return rows


def _candidate_model_search_roots(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hints = _webbduck_path_hints()

    def add(path: Path, source: str) -> None:
        norm_path = path.expanduser().resolve()
        norm = str(norm_path)
        if any(str(item["path"]) == norm for item in rows):
            return
        rows.append({"path": norm_path, "source": source})

    # Preferred conventions aligned to WebbDuck checkpoint roots.
    for checkpoint_wan in _wan_search_roots_from_hints(hints):
        add(checkpoint_wan, "checkpoint_wan")

    # Process cwd fallback.
    add(Path.cwd() / "checkpoint" / "wan", "checkpoint_wan_cwd")
    add(Path.cwd() / "checkpoints" / "wan", "checkpoint_wan_cwd")

    # Plugin-relative repo fallbacks: supports installs inside <webbduck>/plugins/webapps/duckmotion
    # even when the server cwd is different.
    for candidate in _plugin_relative_checkpoint_wan_dirs():
        add(candidate, "checkpoint_wan_plugin")

    configured = _normalize_path_string(config.get("models_dir") or "")
    if configured:
        add(Path(configured), "configured_models_dir")

    for hf_root in _huggingface_cache_roots():
        add(hf_root, "hf_cache")

    return rows


def _plugin_relative_checkpoint_wan_dirs() -> list[Path]:
    rows: list[Path] = []
    seen: set[str] = set()
    for base in [PLUGIN_ROOT, *PLUGIN_ROOT.parents]:
        for rel in [Path("checkpoint") / "wan", Path("checkpoints") / "wan"]:
            candidate = (base / rel).resolve()
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            rows.append(candidate)
    return rows


def _huggingface_cache_roots() -> list[Path]:
    roots: list[Path] = []
    hints = _webbduck_path_hints()
    hinted_hf_cache = hints.get("hf_cache")
    if isinstance(hinted_hf_cache, Path):
        roots.append(hinted_hf_cache)

    env_paths = [
        os.getenv("HUGGINGFACE_HUB_CACHE"),
        os.getenv("HF_HUB_CACHE"),
        os.getenv("TRANSFORMERS_CACHE"),
    ]
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        env_paths.append(str(Path(hf_home).expanduser() / "hub"))

    for raw in env_paths:
        value = _normalize_path_string(raw or "")
        if not value:
            continue
        roots.append(Path(value).expanduser().resolve())

    # Common default cache location.
    roots.append((Path.home() / ".cache" / "huggingface" / "hub").resolve())

    # Deduplicate while preserving order.
    out: list[Path] = []
    seen: set[str] = set()
    for p in roots:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p.resolve())
    return out


def _scan_local_dirs_for_wan_diffusers(root: Path) -> list[Path]:
    found: list[Path] = []

    if _is_wan_diffusers_model_dir(root):
        return [root]

    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        if _is_wan_diffusers_model_dir(child):
            found.append(child)
            continue
        # Allow one nested level to support grouped folders like checkpoint/wan/vendor/model
        try:
            for grand in sorted(child.iterdir(), key=lambda p: p.name.lower()):
                if grand.is_dir() and _is_wan_diffusers_model_dir(grand):
                    found.append(grand)
        except Exception:
            continue
    return found


def _scan_local_files_for_wan_gguf(root: Path) -> list[Path]:
    """Discover standalone quantized Wan transformer files (GGUF and fp8/quantized safetensors)."""
    found: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        found.append(path)

    # GGUF quantized transformer files.
    for pattern in ("*.gguf", "*/*.gguf", "*/*/*.gguf"):
        for candidate in sorted(root.glob(pattern), key=lambda p: p.name.lower()):
            if not candidate.is_file():
                continue
            if "wan" not in candidate.name.lower():
                continue
            add(candidate)

    # Standalone quantized safetensors (fp8, q4, q6, q8) — must NOT be inside a
    # diffusers model dir (those are pipeline component shards, not standalone files).
    _QUANT_HINTS = {"fp8", "fp4", "q4", "q6", "q8", "int8", "int4"}
    for pattern in ("*.safetensors", "*/*.safetensors"):
        for candidate in sorted(root.glob(pattern), key=lambda p: p.name.lower()):
            if not candidate.is_file():
                continue
            name_l = candidate.name.lower()
            if "wan" not in name_l:
                continue
            if not any(h in name_l for h in _QUANT_HINTS):
                continue
            # Skip if it lives inside a diffusers model folder.
            if _is_wan_diffusers_model_dir(candidate.parent):
                continue
            add(candidate)

    return found


def _scan_hf_cache_for_wan_diffusers(hub_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not hub_root.exists() or not hub_root.is_dir():
        return rows

    for model_dir in sorted(hub_root.glob("models--*"), key=lambda p: p.name.lower()):
        if not model_dir.is_dir():
            continue
        name_l = model_dir.name.lower()
        if "wan" not in name_l:
            continue
        repo_id = _repo_id_from_hf_cache_dir_name(model_dir.name)
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists() or not snapshots_dir.is_dir():
            continue
        snapshot_dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        snapshot_dirs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
        for snap in snapshot_dirs:
            if not _is_wan_diffusers_model_dir(snap):
                continue
            label = f"{repo_id or snap.name} (HF cache)"
            if repo_id:
                label = f"{repo_id} (HF cache)"
            rows.append(_discovered_model_item(path=snap, source="hf_cache", repo_id=repo_id, label=label))
            break  # latest usable snapshot only
    return rows


def _repo_id_from_hf_cache_dir_name(name: str) -> str | None:
    raw = str(name or "")
    if not raw.startswith("models--"):
        return None
    return raw[len("models--") :].replace("--", "/").strip() or None


def _repo_id_hint_from_local_name(name: str) -> str | None:
    value = str(name or "").strip()
    if not value:
        return None
    low = value.lower()
    if "wan" not in low:
        return None
    if "diffusers" not in low:
        return None
    return None


def _is_wan_diffusers_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    model_index = path / "model_index.json"
    if not model_index.exists() or not model_index.is_file():
        return False

    # Fast path: name hints.
    name_l = path.name.lower()
    if "wan" in name_l and "diffusers" in name_l:
        return True

    try:
        data = json.loads(model_index.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            class_name = str(data.get("_class_name") or "")
            if "WanImageToVideoPipeline" in class_name:
                return True
            # Broader future-proof Wan match.
            if "Wan" in class_name and "Pipeline" in class_name:
                return True
    except Exception:
        return False
    return False


def _discovered_model_item(*, path: Path, source: str, repo_id: str | None, label: str) -> dict[str, Any]:
    stat_mtime = _safe_mtime(path)
    return {
        "label": str(label or path.name),
        "path": str(path),
        "source": str(source),
        "repo_id": str(repo_id) if repo_id else None,
        "mtime": stat_mtime,
        "is_hf_cache_snapshot": bool(source == "hf_cache"),
        "model_format": "diffusers",
    }


def _discovered_gguf_item(*, path: Path, source: str) -> dict[str, Any]:
    stat_mtime = _safe_mtime(path)
    pairing = _gguf_pairing_info(path.name)
    suffix = path.suffix.lower()
    fmt = "gguf_transformer" if suffix == ".gguf" else "safetensors_transformer"
    return {
        "label": path.name,
        "path": str(path),
        "source": str(source),
        "repo_id": None,
        "mtime": stat_mtime,
        "is_hf_cache_snapshot": False,
        "model_format": fmt,
        "file_format": suffix.lstrip("."),
        "pair_role": pairing["role"],
        "pair_family": pairing["family"],
        "pair_quant": pairing["quant"],
    }


def _resolve_local_diffusers_source(model_source: str, discovered_items: list[dict[str, Any]] | None = None) -> str:
    raw = _normalize_path_string(model_source)
    if not raw:
        return ""
    if _looks_like_path(raw):
        return raw
    items = discovered_items if isinstance(discovered_items, list) else []
    normalized_repo = raw.lower()
    for item in items:
        repo_id = str(item.get("repo_id") or "").strip().lower()
        path = _normalize_path_string(item.get("path") or "")
        if repo_id and repo_id == normalized_repo and path:
            return path
    return raw


def _gguf_pairing_info(filename: str) -> dict[str, str | None]:
    stem = Path(str(filename or "")).stem.strip()
    compact = re.sub(r"\s+", "", stem)
    role_match = re.search(r"([_-]?)(Q\d+[A-Z0-9]*)([_-]?)([HL])$", compact, re.IGNORECASE)
    if not role_match:
        return {"family": stem, "quant": None, "role": None}
    quant = role_match.group(2).upper()
    role = role_match.group(4).upper()
    family = compact[: role_match.start()].rstrip("_-")
    family = re.sub(r"[_-]+$", "", family)
    return {"family": family or stem, "quant": quant, "role": role}


def _find_base_model_for_gguf(gguf_path: str) -> str | None:
    """Look for a companion diffusers model folder alongside a GGUF file."""
    try:
        folder = Path(gguf_path).parent
        for candidate in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
            if candidate.is_dir() and _is_wan_diffusers_model_dir(candidate):
                return str(candidate)
    except Exception:
        pass
    return None


def _group_gguf_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    singles: list[dict[str, Any]] = []
    for item in items:
        family = str(item.get("pair_family") or "").strip()
        quant = str(item.get("pair_quant") or "").strip()
        role = str(item.get("pair_role") or "").strip().upper()
        file_format = str(item.get("file_format") or "gguf")
        if not family or not quant or role not in {"H", "L"}:
            singles.append(
                {
                    "label": str(item.get("label") or item.get("path") or "GGUF"),
                    "family": family or str(item.get("label") or "GGUF"),
                    "quant": quant or None,
                    "file_format": file_format,
                    "paths": {"single": str(item.get("path") or "")},
                    "complete": True,
                    "model_format": f"{file_format}_transformer_single",
                }
            )
            continue
        key = (family, quant, file_format)
        row = grouped.setdefault(
            key,
            {
                "label": f"{family} {quant}",
                "family": family,
                "quant": quant,
                "file_format": file_format,
                "paths": {},
                "complete": False,
                "model_format": f"{file_format}_transformer_pair",
            },
        )
        row["paths"][role] = str(item.get("path") or "")

    out = singles
    for key in sorted(grouped.keys()):
        row = grouped[key]
        row["complete"] = bool(row["paths"].get("H") and row["paths"].get("L"))
        # Attach a companion diffusers base model path if discoverable alongside the GGUF.
        ref_path = row["paths"].get("H") or row["paths"].get("single") or ""
        row["base_model_path"] = _find_base_model_for_gguf(ref_path) if ref_path else None
        out.append(row)
    for row in singles:
        ref_path = row.get("paths", {}).get("single") or ""
        if not row.get("base_model_path") and ref_path:
            row["base_model_path"] = _find_base_model_for_gguf(ref_path)
    out.sort(key=lambda row: str(row.get("label") or "").lower())
    return out


def _discovered_model_sort_key(item: dict[str, Any]) -> tuple[int, str, str]:
    source = str(item.get("source") or "")
    source_rank = {
        "checkpoint_wan": 0,
        "checkpoint_wan_plugin": 1,
        "checkpoint_wan_cwd": 2,
        "configured_models_dir": 3,
        "hf_cache": 4,
    }.get(source, 9)
    label = str(item.get("label") or "").lower()
    path = str(item.get("path") or "").lower()
    return (source_rank, label, path)


def _detect_installation(config: dict[str, Any]) -> dict[str, Any]:
    model_source = _normalize_str(config.get("model_id_or_path") or "")
    models_dir = Path(str(config.get("models_dir") or "").strip()).expanduser() if str(config.get("models_dir") or "").strip() else None
    output_dir = _resolve_output_dir(config)

    model_is_path = _looks_like_path(model_source)
    model_path = Path(model_source).expanduser() if (model_is_path and model_source) else None
    model_path_exists = bool(model_path and model_path.exists())

    diffusers_status = _probe_diffusers_support()
    discovered = _discover_local_models(config)
    gguf_items = discovered.get("gguf_items") if isinstance(discovered, dict) else []
    gguf_candidates = discovered.get("gguf_candidates") if isinstance(discovered, dict) else []
    gguf_transformer_path = _resolve_gguf_transformer_path(
        config,
        gguf_items if isinstance(gguf_items, list) else None,
        gguf_candidates if isinstance(gguf_candidates, list) else None,
    )
    output_writable, output_error = _probe_writable_dir(output_dir)

    return {
        "model_source": model_source,
        "model_source_kind": "path" if model_is_path else ("hf_repo" if model_source else "unset"),
        "model_source_configured": bool(model_source),
        "local_model_path_exists": model_path_exists,
        "models_dir": str(models_dir) if models_dir else "",
        "models_dir_exists": bool(models_dir and models_dir.exists()),
        "runtime_backend": _normalize_runtime_backend(config.get("runtime_backend") or "auto"),
        "gguf_transformer_path": gguf_transformer_path,
        "gguf_transformer_configured": bool(gguf_transformer_path),
        "gguf_transformer_exists": bool(gguf_transformer_path and Path(gguf_transformer_path).expanduser().exists()),
        "gguf_transformer_count": len(gguf_items) if isinstance(gguf_items, list) else 0,
        "gguf_candidate_count": len(gguf_candidates) if isinstance(gguf_candidates, list) else 0,
        "output_dir": str(output_dir),
        "output_dir_writable": output_writable,
        "output_dir_error": output_error,
        "diffusers_ready": bool(diffusers_status.get("ready")),
        "diffusers": diffusers_status,
        "runtime_plan": _plan_runtime_backend(
            config,
            None,
            diffusers_status=diffusers_status,
            discovered_gguf_items=gguf_items if isinstance(gguf_items, list) else None,
            gguf_candidates=gguf_candidates if isinstance(gguf_candidates, list) else None,
        ),
        "missing": _missing_prereqs(model_source, model_is_path, model_path_exists, output_writable, diffusers_status),
    }


def _missing_prereqs(
    model_source: str,
    model_is_path: bool,
    model_path_exists: bool,
    output_writable: bool,
    diffusers_status: dict[str, Any],
) -> list[str]:
    missing: list[str] = []
    if not model_source:
        missing.append("Set a Wan2.2 diffusers model repo ID or local model path")
    elif model_is_path and not model_path_exists:
        missing.append("Configured local model path does not exist")
    if not diffusers_status.get("ready"):
        missing.append(diffusers_status.get("error") or "Install/upgrade diffusers with WanImageToVideoPipeline support")
    if not output_writable:
        missing.append("DuckMotion output directory is not writable")
    return missing


def _probe_diffusers_support() -> dict[str, Any]:
    try:
        import diffusers  # type: ignore
    except Exception as exc:
        return {"ready": False, "error": f"diffusers import failed: {exc}"}

    version = getattr(diffusers, "__version__", "unknown")
    out: dict[str, Any] = {"version": version}
    try:
        from diffusers import WanImageToVideoPipeline  # type: ignore

        cls_name = getattr(WanImageToVideoPipeline, "__name__", "WanImageToVideoPipeline")
        out.update({"ready": True, "pipeline_class": cls_name})
    except Exception as exc:
        out.update({"ready": False, "error": f"WanImageToVideoPipeline unavailable in diffusers ({exc})"})

    try:
        from diffusers import GGUFQuantizationConfig, WanTransformer3DModel  # type: ignore
        import gguf  # type: ignore

        out.update(
            {
                "gguf_ready": True,
                "gguf_version": getattr(gguf, "__version__", "unknown"),
                "gguf_quantization_class": getattr(GGUFQuantizationConfig, "__name__", "GGUFQuantizationConfig"),
                "wan_transformer_class": getattr(WanTransformer3DModel, "__name__", "WanTransformer3DModel"),
                "wan_transformer_single_file": bool(hasattr(WanTransformer3DModel, "from_single_file")),
            }
        )
    except Exception as exc:
        out.update({"gguf_ready": False, "gguf_error": str(exc)})

    return out


def _probe_writable_dir(path: Path) -> tuple[bool, str | None]:
    try:
        path.mkdir(exist_ok=True, parents=True)
        probe = path / f".duckmotion_write_test_{uuid.uuid4().hex}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _engine_status(config: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    queue_snapshot = _queue_snapshot()
    pipeline_loaded = _PIPELINE is not None
    runtime_profile = runtime.get("profile") if isinstance(runtime, dict) else None
    memory_strategy = _PIPELINE_MEMORY_STRATEGY
    if memory_strategy is None and isinstance(runtime_profile, dict):
        memory_strategy = _plan_memory_strategy(config, runtime_profile)
    installation = _detect_installation(config)
    runtime_plan = _plan_runtime_backend(
        config,
        runtime_profile if isinstance(runtime_profile, dict) else None,
        diffusers_status=installation.get("diffusers") if isinstance(installation, dict) else None,
        discovered_gguf_items=_discover_local_models(config).get("gguf_items"),
    )
    return {
        "type": "local-diffusers",
        "runtime_profile": runtime,
        "gpu_lease": get_gpu_lease(),
        "queue": queue_snapshot,
        "pipeline_loaded": pipeline_loaded,
        "model_source": _normalize_str(config.get("model_id_or_path") or ""),
        "output_dir": str(_resolve_output_dir(config)),
        "memory_strategy": memory_strategy,
        "runtime_plan": runtime_plan,
    }


# ---------------------------------------------------------------------------
# WebbDuck runtime profile reuse
# ---------------------------------------------------------------------------


def _get_runtime_profile_safe() -> dict[str, Any]:
    try:
        from webbduck.core.runtime import resolve_runtime_profile  # lazy import

        profile = resolve_runtime_profile()
        return {"ok": True, "profile": profile.to_dict(), "source": "webbduck.core.runtime.resolve_runtime_profile"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _get_runtime_profile_or_raise() -> dict[str, Any]:
    runtime = _get_runtime_profile_safe()
    if not runtime.get("ok"):
        raise RuntimeError(f"Runtime profile unavailable: {runtime.get('error')}")
    profile = runtime.get("profile")
    if not isinstance(profile, dict):
        raise RuntimeError("Runtime profile payload missing.")
    return profile


# ---------------------------------------------------------------------------
# Job queue + persistence
# ---------------------------------------------------------------------------


def _ensure_worker_started() -> None:
    global _WORKER_THREAD
    with _LOCK:
        if _WORKER_THREAD is not None and _WORKER_THREAD.is_alive():
            return
        _recover_incomplete_jobs_locked()
        _WORKER_THREAD = threading.Thread(target=_worker_loop, name="duckmotion-worker", daemon=True)
        _WORKER_THREAD.start()


def _worker_loop() -> None:
    while True:
        job_id = _JOB_QUEUE.get()
        try:
            _run_job(job_id)
        except Exception:
            _mark_job_failed(job_id, "Unhandled worker error.")
        finally:
            _JOB_QUEUE.task_done()


def _queue_snapshot() -> dict[str, Any]:
    with _LOCK:
        jobs = _load_jobs_locked()
        queued = sum(1 for j in jobs if j.get("status") == "queued")
        running = sum(1 for j in jobs if j.get("status") == "running")
        cancel_requested = sum(1 for j in jobs if j.get("cancel_requested"))
    return {"queued": queued, "running": running, "cancel_requested": cancel_requested}


def _load_jobs_locked() -> list[dict[str, Any]]:
    if not JOBS_FILE.exists():
        return []
    try:
        data = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    rows = [row for row in data if isinstance(row, dict)]
    return rows[-500:]


def _recover_incomplete_jobs_locked() -> None:
    rows = _load_jobs_locked()
    if not rows:
        return

    changed = False
    ts = _now()
    for row in rows:
        status = str(row.get("status") or "").strip().lower()
        if status not in {"running", "cancel_requested"}:
            continue
        changed = True
        if status == "cancel_requested":
            row["status"] = "canceled"
            row["error"] = "Canceled after runtime restart."
            row["progress"] = {"stage": "canceled", "percent": row.get("progress", {}).get("percent", 0)}
        else:
            row["status"] = "failed"
            row["error"] = "Interrupted by runtime restart while job was running."
            row["progress"] = {"stage": "failed", "percent": row.get("progress", {}).get("percent", 0)}
        row["updated_at"] = ts
        row["finished_at"] = ts
        row["cancel_requested"] = False

    if changed:
        _save_jobs_locked(rows)


def _save_jobs_locked(rows: list[dict[str, Any]]) -> None:
    STATE_DIR.mkdir(exist_ok=True, parents=True)
    JOBS_FILE.write_text(json.dumps(rows[-500:], indent=2), encoding="utf-8")


def _list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with _LOCK:
        rows = _load_jobs_locked()
    rows.sort(key=lambda r: float(r.get("created_at", 0.0)), reverse=True)
    return rows[:limit]


def _get_job(job_id: str) -> dict[str, Any] | None:
    target = str(job_id or "").strip()
    if not target:
        return None
    with _LOCK:
        rows = _load_jobs_locked()
    for row in rows:
        if str(row.get("job_id") or "") == target:
            return row
    return None


def _upsert_job(job: dict[str, Any]) -> dict[str, Any]:
    with _LOCK:
        rows = _load_jobs_locked()
        idx = None
        for i, row in enumerate(rows):
            if str(row.get("job_id") or "") == str(job.get("job_id") or ""):
                idx = i
                break
        if idx is None:
            rows.append(job)
        else:
            merged = dict(rows[idx])
            merged.update(job)
            rows[idx] = merged
            job = merged
        _save_jobs_locked(rows)
    return job


def _submit_job(payload: GeneratePayload, config: dict[str, Any]) -> dict[str, Any]:
    src_path = _resolve_input_image(payload.image_path)
    now = _now()
    job_id = f"dm_{uuid.uuid4().hex[:12]}"
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "cancel_requested": False,
        "progress": {"stage": "queued", "percent": 0},
        "config_snapshot": {
            "model_id_or_path": _normalize_str(config.get("model_id_or_path") or ""),
            "output_dir": str(_resolve_output_dir(config)),
        },
        "input": {
            "image_path": str(src_path),
            "image_name": src_path.name,
            "web_path": _safe_web_path_for_any(src_path),
        },
        "params": _normalize_generate_params(payload, config),
        "error": None,
        "warnings": [],
        "outputs": [],
    }
    _upsert_job(job)
    _JOB_QUEUE.put(job_id)
    return job


def _cancel_job(job_id: str) -> dict[str, Any]:
    row = _get_job(job_id)
    if row is None:
        return {"ok": False, "error": "Job not found."}
    status = str(row.get("status") or "")
    if status in {"completed", "failed", "canceled"}:
        return {"ok": True, "job": row, "message": f"Job already {status}."}

    row["cancel_requested"] = True
    row["updated_at"] = _now()
    if status == "queued":
        row["status"] = "canceled"
        row["finished_at"] = _now()
        row["progress"] = {"stage": "canceled", "percent": 0}
    else:
        row["status"] = "cancel_requested"
        row["progress"] = {"stage": "cancel_requested", "percent": row.get("progress", {}).get("percent", 0)}
    row = _upsert_job(row)
    return {
        "ok": True,
        "job": row,
        "message": "Cancellation requested. Running diffusers jobs may complete before the request can take effect.",
    }


def _mark_job_failed(job_id: str, message: str) -> None:
    row = _get_job(job_id)
    if row is None:
        return
    row["status"] = "failed"
    row["error"] = str(message)
    row["updated_at"] = _now()
    row["finished_at"] = _now()
    row["progress"] = {"stage": "failed", "percent": row.get("progress", {}).get("percent", 0)}
    _upsert_job(row)


# ---------------------------------------------------------------------------
# Generation worker (diffusers)
# ---------------------------------------------------------------------------


def _run_job(job_id: str) -> None:
    row = _get_job(job_id)
    if row is None:
        return
    if row.get("status") == "canceled":
        return
    if row.get("cancel_requested") and row.get("status") == "queued":
        row["status"] = "canceled"
        row["updated_at"] = _now()
        row["finished_at"] = _now()
        row["progress"] = {"stage": "canceled", "percent": 0}
        _upsert_job(row)
        return

    row["status"] = "running"
    row["started_at"] = _now()
    row["updated_at"] = _now()
    row["progress"] = {"stage": "preparing", "percent": 5}
    _upsert_job(row)

    lease_token = None
    config: dict[str, Any] = {}
    try:
        config = _load_config()
        runtime_profile = _get_runtime_profile_or_raise()
        row["runtime_profile"] = runtime_profile
        safety = _evaluate_job_safety(row.get("params") or {}, config, runtime_profile)
        row["safety"] = safety
        if not safety.get("ok"):
            raise RuntimeError(str(safety.get("message") or "DuckMotion safety gate blocked this job."))
        row["progress"] = {"stage": "waiting_for_gpu", "percent": 10}
        row["updated_at"] = _now()
        _upsert_job(row)

        lease_attempt = acquire_gpu_lease_blocking(
            owner="duckmotion",
            owner_kind="plugin",
            label="wan_i2v",
            job_id=job_id,
        )
        lease = lease_attempt.get("lease") if isinstance(lease_attempt, dict) else None
        if isinstance(lease, dict):
            lease_token = str(lease.get("token") or "").strip() or None

        row["progress"] = {"stage": "preparing_gpu", "percent": 10}
        row["updated_at"] = _now()
        _upsert_job(row)
        _prepare_runtime_for_wan(runtime_profile)

        row["progress"] = {"stage": "loading_pipeline", "percent": 15}
        row["updated_at"] = _now()
        _upsert_job(row)

        if row.get("cancel_requested"):
            row["status"] = "canceled"
            row["finished_at"] = _now()
            row["updated_at"] = _now()
            row["progress"] = {"stage": "canceled", "percent": 15}
            _upsert_job(row)
            return

        if _should_isolate_process():
            output_info = _run_generation_isolated(row, config, runtime_profile)
        else:
            frames = _generate_frames_with_diffusers(row, config, runtime_profile)

            row = _get_job(job_id) or row
            row["progress"] = {"stage": "writing_outputs", "percent": 85}
            row["updated_at"] = _now()
            _upsert_job(row)

            output_info = _write_video_outputs(row, frames, config)

        row = _get_job(job_id) or row
        row["status"] = "completed"
        row["error"] = None
        row["outputs"] = [output_info]
        row["updated_at"] = _now()
        row["finished_at"] = _now()
        row["progress"] = {"stage": "completed", "percent": 100}
        _upsert_job(row)
    except Exception as exc:
        err_msg = str(exc or "").strip()
        if not err_msg:
            err_msg = f"{exc.__class__.__name__} (no exception message)"
        if _is_missing_accelerate_error(exc):
            err_msg = (
                "DuckMotion offload mode requires `accelerate`. "
                "Install/upgrade `accelerate` in the WebbDuck environment, then retry."
            )
        elif _is_low_cpu_mem_usage_fp32_constraint_error(exc):
            err_msg = (
                "Incompatible diffusers loading flags for this Wan model "
                "(`low_cpu_mem_usage=False` with fp32-keep modules). "
                "Restart WebbDuck to pick up the latest DuckMotion loader fix and retry."
            )
        elif _is_meta_tensor_move_error(exc):
            err_msg = (
                "Wan pipeline CUDA placement failed due to meta tensors. "
                "Restart WebbDuck and retry; if it persists, keep DUCKMOTION_CUDA_MODE=offload "
                "and install/upgrade diffusers + accelerate."
            )
        if _is_windows_paging_file_error(exc):
            err_msg = (
                "Windows virtual memory (page file) is too small to load this Wan checkpoint. "
                "Increase page file size (recommend 64-128 GB total), reboot Windows, then retry."
            )
        _LOG.exception("DuckMotion job %s failed", job_id)
        try:
            from webbduck.core.runtime import runtime_error_hint  # type: ignore

            hint = runtime_error_hint(exc)
        except Exception:
            hint = None

        row = _get_job(job_id) or row
        row["status"] = "failed"
        row["error"] = err_msg
        if hint:
            warnings = row.get("warnings") or []
            if isinstance(warnings, list):
                warnings = [*warnings, hint]
                row["warnings"] = warnings[-8:]
        if _is_windows_paging_file_error(exc):
            warnings = row.get("warnings") or []
            if isinstance(warnings, list):
                warnings = [
                    *warnings,
                    "System > Advanced system settings > Performance > Advanced > Virtual memory: increase paging file.",
                    "If possible, use System managed size or set a custom size around 65536-131072 MB.",
                ]
                row["warnings"] = warnings[-8:]
        if _is_missing_accelerate_error(exc):
            warnings = row.get("warnings") or []
            if isinstance(warnings, list):
                warnings = [
                    *warnings,
                    "Install plugin runtime deps in WebbDuck env: pip install -r DuckMotion/requirements.txt",
                    "Or switch to explicit full CUDA mode: DUCKMOTION_CUDA_MODE=full (higher VRAM/RAM use).",
                ]
                row["warnings"] = warnings[-8:]
        row["updated_at"] = _now()
        row["finished_at"] = _now()
        row["progress"] = {"stage": "failed", "percent": row.get("progress", {}).get("percent", 0)}
        _upsert_job(row)
    finally:
        if not _should_keep_pipeline_loaded(config):
            _unload_pipeline()
        if lease_token:
            try:
                release_gpu_lease(token=lease_token)
            except Exception:
                pass


def _run_generation_isolated(job: dict[str, Any], config: dict[str, Any], runtime_profile: dict[str, Any]) -> dict[str, Any]:
    STATE_DIR.mkdir(exist_ok=True, parents=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="duckmotion_child_", dir=str(STATE_DIR)))
    payload_path = tmp_dir / "payload.json"
    result_path = tmp_dir / "result.json"
    progress_path = tmp_dir / "progress.json"
    log_path = tmp_dir / "child.log"
    job_id = str(job.get("job_id") or "unknown")
    try:
        payload = {
            "job": job,
            "config": config,
            "runtime_profile": runtime_profile,
            "progress_path": str(progress_path),
        }
        payload_path.write_text(json.dumps(payload), encoding="utf-8")

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--duckmotion-child",
            "--payload",
            str(payload_path),
            "--result",
            str(result_path),
        ]
        # Use a short timeout for model loading, but once generation starts allow
        # a much longer window because Wan denoising can legitimately take time.
        try:
            with open(str(log_path), "w", encoding="utf-8") as log_f:
                child_proc = subprocess.Popen(cmd, text=True, stdout=log_f, stderr=subprocess.STDOUT)
                current_stage = "loading_pipeline"
                deadline = _now() + 300
                last_progress: tuple[str, int] | None = None
                return_code = None
                while True:
                    return_code = child_proc.poll()
                    if progress_path.exists():
                        try:
                            raw_progress = json.loads(progress_path.read_text(encoding="utf-8"))
                        except Exception:
                            raw_progress = None
                        if isinstance(raw_progress, dict):
                            stage = str(raw_progress.get("stage") or "").strip()
                            try:
                                percent = int(raw_progress.get("percent"))
                            except Exception:
                                percent = -1
                            marker = (stage, percent)
                            if stage and percent >= 0 and marker != last_progress:
                                current_stage = stage
                                row = _get_job(job_id) or job
                                detail = str(raw_progress.get("detail") or "").strip()
                                row["progress"] = {"stage": stage, "percent": percent}
                                if detail:
                                    row["progress"]["detail"] = detail
                                row["updated_at"] = _now()
                                _upsert_job(row)
                                last_progress = marker
                                # Extend the timeout substantially once actual generation/output
                                # work begins; the 5-minute limit should only guard stalled loads.
                                if stage in {"generating", "writing_outputs", "completed"}:
                                    deadline = _now() + 1800
                                else:
                                    deadline = _now() + 300
                    if return_code is not None:
                        break
                    if _now() >= deadline:
                        child_proc.kill()
                        child_proc.wait(timeout=10)
                        timeout_window = 1800 if current_stage in {"generating", "writing_outputs", "completed"} else 300
                        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_window)
                    time.sleep(0.5)
                proc = child_proc
        except subprocess.TimeoutExpired as exc:
            log_detail = ""
            if log_path.exists():
                log_text = log_path.read_text(encoding="utf-8")
                log_detail = f" Child log (last 800 chars): {log_text[-800:]}" if log_text else ""
            timeout_seconds = 300
            try:
                timeout_seconds = int(getattr(exc, "timeout", 300) or 300)
            except Exception:
                timeout_seconds = 300
            timeout_minutes = max(1, timeout_seconds // 60)
            raise RuntimeError(
                f"DuckMotion isolated runtime timed out after {timeout_minutes} minutes. "
                f"This usually indicates the model is still loading or has stalled.{log_detail}"
            )

        if proc.returncode != 0 and not result_path.exists():
            log_detail = ""
            if log_path.exists():
                log_text = log_path.read_text(encoding="utf-8")
                log_detail = f" Child log (last 800 chars): {log_text[-800:]}" if log_text else ""
            crash_hint = ""
            if _is_windows_access_violation_rc(proc.returncode):
                crash_hint = (
                    " Windows access-violation crash detected (often memory pressure/page-file exhaustion during model load). "
                    "Try DUCKMOTION_CUDA_MODE=full so the GPU is used earlier, and lower load settings (for example: 576x320, "
                    "49 frames, 20 steps)."
                )
            raise RuntimeError(
                "DuckMotion isolated runtime exited unexpectedly "
                f"(code {proc.returncode}). This usually indicates host memory pressure or a runtime crash.{crash_hint}{log_detail}"
            )

        if not result_path.exists():
            log_detail = ""
            if log_path.exists():
                log_text = log_path.read_text(encoding="utf-8")
                log_detail = f" Child log (last 800 chars): {log_text[-800:]}" if log_text else ""
            raise RuntimeError(f"DuckMotion isolated runtime exited without a result payload.{log_detail}")

        raw = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RuntimeError("DuckMotion isolated runtime returned invalid result payload.")
        if not raw.get("ok"):
            msg = str(raw.get("error") or "").strip() or "DuckMotion isolated runtime failed."
            raise RuntimeError(msg)

        output_info = raw.get("output_info")
        if not isinstance(output_info, dict):
            raise RuntimeError("DuckMotion isolated runtime did not return output metadata.")
        return output_info
    except Exception:
        # Preserve the child log for debugging when an error occurs.
        if log_path.exists():
            persist_log = STATE_DIR / f"duckmotion_child_fail_{job_id}.log"
            try:
                persist_log.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass
        raise
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def _run_generation_child(payload_path: Path, result_path: Path) -> int:
    try:
        payload_raw = json.loads(payload_path.read_text(encoding="utf-8"))
        if not isinstance(payload_raw, dict):
            raise RuntimeError("Invalid child payload.")
        job = payload_raw.get("job")
        config = payload_raw.get("config")
        runtime_profile = payload_raw.get("runtime_profile")
        progress_path_raw = payload_raw.get("progress_path")
        if not isinstance(job, dict) or not isinstance(config, dict) or not isinstance(runtime_profile, dict):
            raise RuntimeError("Child payload missing required fields.")
        progress_path = Path(str(progress_path_raw)).expanduser() if progress_path_raw else None

        _write_progress_file(progress_path, "loading_pipeline", 15)
        frames = _generate_frames_with_diffusers(job, config, runtime_profile, persist_progress=False, progress_path=progress_path)
        _write_progress_file(progress_path, "writing_outputs", 90)
        output_info = _write_video_outputs(job, frames, config)
        payload = {"ok": True, "output_info": output_info}
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return 0
    except Exception as exc:
        payload = {
            "ok": False,
            "error": str(exc or "").strip() or f"{exc.__class__.__name__} (no message)",
            "traceback": traceback.format_exc(),
        }
        try:
            result_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass
        return 1


def _generate_frames_with_diffusers(
    job: dict[str, Any],
    config: dict[str, Any],
    runtime_profile: dict[str, Any],
    *,
    persist_progress: bool = True,
    progress_path: Path | None = None,
) -> list[Any]:
    params = job.get("params") or {}
    if not isinstance(params, dict):
        raise RuntimeError("Invalid job parameters.")

    source = Path(str((job.get("input") or {}).get("image_path") or ""))
    if not source.exists() or not source.is_file():
        raise RuntimeError("Input image not found.")

    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Required runtime dependency missing: {exc}") from exc

    _write_progress_file(progress_path, "loading_pipeline", 20)
    pipe = _get_or_load_pipeline(config, runtime_profile, progress_path=progress_path)

    image = Image.open(source).convert("RGB")
    width = int(params.get("width") or image.width)
    height = int(params.get("height") or image.height)
    width, height = _snap_dimensions(width, height)
    if (width, height) != (image.width, image.height):
        image = image.resize((width, height))

    seed = params.get("seed")
    device = str(runtime_profile.get("device") or "cpu")
    generator = None
    if seed is not None:
        try:
            generator = torch.Generator(device=device).manual_seed(int(seed))
        except Exception:
            generator = torch.Generator().manual_seed(int(seed))

    prompt = str(params.get("prompt") or "").strip()
    if not prompt:
        raise RuntimeError("Prompt is required.")

    call_kwargs = {
        "image": image,
        "prompt": prompt,
        "negative_prompt": str(params.get("negative_prompt") or ""),
        "height": height,
        "width": width,
        "num_frames": int(params.get("num_frames") or 81),
        "guidance_scale": float(params.get("guidance_scale") or 5.0),
        "num_inference_steps": int(params.get("num_inference_steps") or 30),
    }
    if generator is not None:
        call_kwargs["generator"] = generator

    total_steps = max(1, int(params.get("num_inference_steps") or 30))
    try:
        call_sig = inspect.signature(pipe.__call__)
    except Exception:
        call_sig = None
    if call_sig and "callback_on_step_end" in call_sig.parameters:
        def _progress_callback(_pipe: Any, step_index: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            step_num = max(0, int(step_index)) + 1
            pct = min(88, 60 + int((step_num / total_steps) * 28))
            _write_progress_file(progress_path, "generating", pct, detail=f"Step {step_num}/{total_steps}")
            return callback_kwargs

        call_kwargs["callback_on_step_end"] = _progress_callback
        if "callback_on_step_end_tensor_inputs" in call_sig.parameters:
            call_kwargs["callback_on_step_end_tensor_inputs"] = []

    if persist_progress:
        row = _get_job(str(job.get("job_id") or "")) or job
        row["progress"] = {"stage": "generating", "percent": 40}
        row["updated_at"] = _now()
        _upsert_job(row)

    _write_progress_file(progress_path, "generating", 60, detail=f"Step 0/{total_steps}")

    result = pipe(**call_kwargs)
    frames = _extract_frames_from_result(result)
    if not frames:
        raise RuntimeError("Diffusers pipeline returned no video frames.")
    return frames


def _extract_frames_from_result(result: Any) -> list[Any]:
    frames = getattr(result, "frames", None)
    if frames is None and isinstance(result, tuple) and result:
        frames = result[0]
    if frames is None:
        return []
    if isinstance(frames, list):
        if not frames:
            return []
        if isinstance(frames[0], list):
            return list(frames[0])
        return list(frames)
    if hasattr(frames, "ndim"):
        try:
            ndim = int(frames.ndim)
        except Exception:
            ndim = -1
        if ndim == 5:
            return list(frames[0])
        if ndim == 4:
            return list(frames)
    if isinstance(frames, tuple):
        return list(frames)
    return []


def _normalize_frame_to_pil(frame: Any) -> Any:
    from PIL import Image  # type: ignore

    if isinstance(frame, Image.Image):
        return frame

    if hasattr(frame, "detach") and hasattr(frame, "cpu") and hasattr(frame, "numpy"):
        frame = frame.detach().cpu().numpy()

    if hasattr(frame, "ndim") and hasattr(frame, "dtype"):
        try:
            import numpy as np  # type: ignore

            arr = frame
            if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
                arr = np.transpose(arr, (1, 2, 0))
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0).round().astype("uint8")
            elif arr.dtype != np.uint8:
                arr = arr.astype("uint8")
            return Image.fromarray(arr)
        except Exception:
            return frame

    return frame


def _normalize_frames_to_pil(frames: list[Any]) -> list[Any]:
    return [_normalize_frame_to_pil(frame) for frame in frames]


def _load_gguf_pipeline(
    model_source: str,
    gguf_transformer_path: str,
    gguf_pair_paths: dict[str, str],
    dtype: Any,
    device: str,
    cache_dir: str,
    model_source_exists: bool,
    progress_path: Path | None = None,
    use_device_map_cuda: bool = True,
) -> Any:
    """Build a WanImageToVideoPipeline using a quantized transformer file.

    Supports two formats transparently:
    - .gguf  → GGUFQuantizationConfig + WanTransformer3DModel.from_single_file
    - .safetensors → WanTransformer3DModel.from_single_file directly (fp8 / quantized weights)

    In both cases the base diffusers repo provides the config, scheduler, VAE and text encoders.
    The quantized transformer is injected as a drop-in replacement, keeping the pipeline API
    identical to the full diffusers path.
    """
    try:
        import torch  # type: ignore
        from diffusers import WanImageToVideoPipeline, WanTransformer3DModel  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"DuckMotion quantized backend requires diffusers with WanTransformer3DModel: {exc}") from exc

    suffix = Path(gguf_transformer_path).suffix.lower()
    is_gguf = suffix == ".gguf"

    _LOG.info(
        "DuckMotion loading %s transformer: %s (H=%s L=%s)",
        "GGUF" if is_gguf else "safetensors",
        gguf_transformer_path,
        gguf_pair_paths.get("H", ""),
        gguf_pair_paths.get("L", ""),
    )

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if is_gguf:
        try:
            from diffusers import GGUFQuantizationConfig  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"DuckMotion GGUF loading requires GGUFQuantizationConfig in diffusers: {exc}") from exc
        load_kwargs["quantization_config"] = GGUFQuantizationConfig(compute_dtype=dtype)

    # For GGUF/safetensors single-file loading, the config must come from a
    # diffusers repo (HF cache or local folder). Use the HF repo ID as the
    # source since the GGUF file itself does not contain a config.json.
    gguf_config_source = model_source if (model_source_exists and not is_gguf and suffix != ".safetensors") else DEFAULT_MODEL_ID
    load_kwargs["config"] = gguf_config_source
    load_kwargs["subfolder"] = "transformer"
    if cache_dir and not model_source_exists:
        load_kwargs["cache_dir"] = str(Path(cache_dir).expanduser())

    try:
        _write_progress_file(progress_path, "loading_transformer", 25)
        transformer = WanTransformer3DModel.from_single_file(gguf_transformer_path, **load_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"DuckMotion failed to load quantized transformer from {gguf_transformer_path}: {exc}"
        ) from exc

    _LOG.info("DuckMotion quantized transformer loaded; assembling pipeline from base repo.")

    # For GGUF/safetensors pipelines, use the HF repo ID as the base source since
    # the single-file transformer does not include pipeline components (VAE, text encoders, etc.).
    pipe_base_source = model_source if model_source_exists else DEFAULT_MODEL_ID
    pipe_kwargs: dict[str, Any] = {
        "transformer": transformer,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda" and use_device_map_cuda:
        pipe_kwargs["device_map"] = "cuda"
    if cache_dir and not model_source_exists:
        pipe_kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
    if model_source_exists:
        pipe_kwargs["local_files_only"] = True

    try:
        _write_progress_file(progress_path, "assembling_pipeline", 40)
        pipe = WanImageToVideoPipeline.from_pretrained(pipe_base_source, **pipe_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"DuckMotion failed to assemble pipeline from base repo '{model_source}' with quantized transformer: {exc}"
        ) from exc

    return pipe


def _get_or_load_pipeline(config: dict[str, Any], runtime_profile: dict[str, Any], progress_path: Path | None = None) -> Any:
    global _PIPELINE, _PIPELINE_KEY, _PIPELINE_MEMORY_STRATEGY
    model_source = _normalize_str(config.get("model_id_or_path") or "")
    if model_source.startswith("gguf:"):
        model_source = model_source[5:]
    if not model_source:
        raise RuntimeError("DuckMotion model_id_or_path is not configured.")

    dtype_name = str(runtime_profile.get("dtype") or "float32")
    device = str(runtime_profile.get("device") or "cpu")
    cache_dir = _normalize_path_string(config.get("models_dir") or "")

    discovered = _discover_local_models(config)
    discovered_items = discovered.get("items") if isinstance(discovered.get("items"), list) else []
    runtime_plan = _plan_runtime_backend(
        config,
        runtime_profile,
        diffusers_status=_probe_diffusers_support(),
        discovered_gguf_items=discovered.get("gguf_items"),
        gguf_candidates=discovered.get("gguf_candidates"),
    )
    selected_backend = runtime_plan.get("selected_backend") or "diffusers"
    gguf_candidate = runtime_plan.get("gguf_candidate") or {}
    gguf_pair_paths = gguf_candidate.get("paths") or {}
    gguf_transformer_path = _normalize_path_string(runtime_plan.get("gguf_transformer_path") or "")
    effective_model_source = (
        _normalize_path_string(runtime_plan.get("gguf_base_model_path") or "") if selected_backend == "diffusers_gguf" else model_source
    )
    if not effective_model_source:
        effective_model_source = model_source
    effective_model_source = _resolve_local_diffusers_source(effective_model_source, discovered_items)
    effective_model_source_is_path = _looks_like_path(effective_model_source)
    effective_model_source_path = Path(effective_model_source).expanduser() if effective_model_source_is_path else None
    effective_model_source_exists = bool(effective_model_source_path and effective_model_source_path.exists())

    key = (effective_model_source, device, dtype_name, selected_backend, gguf_transformer_path)

    with _PIPELINE_LOCK:
        if _PIPELINE is not None and _PIPELINE_KEY == key:
            return _PIPELINE

        try:
            import torch  # type: ignore
            from diffusers import WanImageToVideoPipeline  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "DuckMotion requires a diffusers build with WanImageToVideoPipeline. "
                f"Import failed: {exc}"
            ) from exc

        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            dtype = torch.float32
        if os.name == "nt" and device == "cuda" and dtype is not torch.float16:
            # Wan + diffusers on Windows is generally more stable and memory-safer in fp16.
            dtype = torch.float16
            _LOG.warning("DuckMotion overriding runtime dtype to float16 for Wan pipeline on Windows CUDA.")

        def _load_pipeline(*, low_mem: bool) -> Any:
            base_load_kwargs: dict[str, Any] = {}
            # Only apply cache_dir when loading from remote repo ID.
            # For local model paths, passing cache_dir can trigger unnecessary HF cache traversal.
            if cache_dir and not effective_model_source_exists:
                base_load_kwargs["cache_dir"] = str(Path(cache_dir).expanduser())

            if low_mem:
                # Lower host-memory load path first (important on Windows with large Wan checkpoints).
                load_attempts = [
                    {"torch_dtype": dtype, "low_cpu_mem_usage": True, "offload_state_dict": False},
                    {"torch_dtype": dtype, "low_cpu_mem_usage": True, "offload_state_dict": True},
                    {"torch_dtype": dtype},
                ]
            else:
                # Compatibility/materialized-leaning path for explicit full-CUDA placement.
                # Do NOT set low_cpu_mem_usage=False for Wan configs that keep fp32 modules.
                load_attempts = [
                    {"torch_dtype": dtype, "low_cpu_mem_usage": True, "offload_state_dict": False},
                    {"torch_dtype": dtype, "low_cpu_mem_usage": True},
                    {"torch_dtype": dtype},
                ]

            if _allow_unsafe_load_fallbacks():
                # Optional compatibility fallback path (can significantly increase host RAM usage).
                if low_mem:
                    load_attempts.extend(
                        [
                            {"low_cpu_mem_usage": True, "offload_state_dict": False},
                            {"low_cpu_mem_usage": True, "offload_state_dict": True},
                            {},
                        ]
                    )
                else:
                    load_attempts.extend(
                        [
                            {"low_cpu_mem_usage": True, "offload_state_dict": False},
                            {"low_cpu_mem_usage": True},
                            {},
                        ]
                    )

            load_errors: list[Exception] = []
            for attempt in load_attempts:
                kwargs = dict(base_load_kwargs)
                kwargs.update(attempt)
                try:
                    _write_progress_file(progress_path, "assembling_pipeline", 35)
                    if effective_model_source_exists:
                        kwargs["local_files_only"] = True
                    return WanImageToVideoPipeline.from_pretrained(effective_model_source, **kwargs)
                except TypeError as exc:
                    # Compatibility fallback: older/newer diffusers signatures differ.
                    load_errors.append(exc)
                    continue
                except Exception:
                    raise
            if load_errors:
                raise load_errors[-1]
            raise RuntimeError("Failed to load WanImageToVideoPipeline from model source.")

        cuda_mode = _resolve_effective_cuda_mode(config, runtime_profile, selected_backend)
        # For GGUF backends the memory policy is forced to aggressive because the
        # quantized transformer is already lighter; we want every other optimization
        # (attention/VAE slicing, sequential offload) applied automatically too.
        if selected_backend == "diffusers_gguf":
            gguf_config = dict(config)
            gguf_config["cuda_mode"] = cuda_mode
            if _normalize_memory_policy(gguf_config.get("memory_policy") or "auto") == "off":
                gguf_config["memory_policy"] = "balanced"
            memory_strategy = _plan_memory_strategy(gguf_config, runtime_profile)
        else:
            plain_config = dict(config)
            plain_config["cuda_mode"] = cuda_mode
            memory_strategy = _plan_memory_strategy(plain_config, runtime_profile)
        effective_offload_strategy = _resolve_effective_offload_strategy(memory_strategy, runtime_profile, selected_backend)
        if effective_offload_strategy != str(memory_strategy.get("offload_strategy") or "none"):
            memory_strategy = dict(memory_strategy)
            memory_strategy["offload_strategy"] = effective_offload_strategy
            notes = list(memory_strategy.get("notes") or [])
            notes.append(f"selected {effective_offload_strategy} for {selected_backend} on this VRAM tier")
            memory_strategy["notes"] = notes

        if selected_backend == "diffusers_gguf" and gguf_transformer_path:
            _LOG.info(
                "DuckMotion using diffusers_gguf backend (H=%s L=%s)",
                gguf_pair_paths.get("H", ""),
                gguf_pair_paths.get("L", ""),
            )
            def _build_gguf_pipe(*, use_device_map_cuda: bool) -> Any:
                return _load_gguf_pipeline(
                    model_source=effective_model_source,
                    gguf_transformer_path=gguf_transformer_path,
                    gguf_pair_paths=gguf_pair_paths,
                    dtype=dtype,
                    device=device,
                    cache_dir=cache_dir,
                    model_source_exists=effective_model_source_exists,
                    progress_path=progress_path,
                    use_device_map_cuda=use_device_map_cuda,
                )

            pipe = _build_gguf_pipe(use_device_map_cuda=(device == "cuda"))
        else:
            pipe = _load_pipeline(low_mem=(device != "cuda" or cuda_mode != "full"))

        _write_progress_file(progress_path, "pipeline_loaded", 50)

        if device == "cuda":
            placed = False
            if cuda_mode != "full":
                try:
                    if getattr(pipe, "hf_device_map", None):
                        reset_device_map = getattr(pipe, "reset_device_map", None)
                        if callable(reset_device_map):
                            reset_device_map()
                    _enable_cuda_offload(pipe, memory_strategy)
                    placed = True
                except Exception as exc:
                    _LOG.warning("DuckMotion CPU offload init failed: %s", exc)
                    if _is_missing_accelerate_error(exc):
                        raise RuntimeError(
                            "DuckMotion offload mode requires `accelerate`. "
                            "Install `accelerate` in the WebbDuck environment, or set DUCKMOTION_CUDA_MODE=full."
                        ) from exc
                    if _is_meta_tensor_move_error(exc):
                        raise RuntimeError(
                            "DuckMotion offload init hit a meta-tensor move issue. "
                            "Use WEBBDUCK_DTYPE=float16 and DUCKMOTION_CUDA_MODE=offload, then restart WebbDuck."
                        ) from exc
                    raise RuntimeError(
                        "DuckMotion failed to initialize CPU offload mode. "
                        "Restart WebbDuck and retry, or set DUCKMOTION_CUDA_MODE=full."
                    ) from exc

            if not placed:
                try:
                    if getattr(pipe, "hf_device_map", None):
                        reset_device_map = getattr(pipe, "reset_device_map", None)
                        if callable(reset_device_map):
                            reset_device_map()
                    pipe = pipe.to("cuda")
                    placed = True
                except Exception as exc:
                    if _is_meta_tensor_move_error(exc):
                        raise RuntimeError(
                            "Wan pipeline CUDA placement failed due to meta tensors. "
                            "Use DUCKMOTION_CUDA_MODE=offload and WEBBDUCK_DTYPE=float16."
                        ) from exc

                    if _is_cuda_oom(exc):
                        _LOG.warning("CUDA OOM while loading Wan pipeline; attempting CPU offload fallback.")
                        _write_progress_file(progress_path, "fallback_offload", 52, detail="CUDA OOM; rebuilding with CPU offload")
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        if not hasattr(pipe, "enable_model_cpu_offload"):
                            raise RuntimeError(
                                "CUDA out of memory while loading the Wan pipeline. "
                                "Install `accelerate` and use CPU offload, or reduce model size/resolution/frames."
                            ) from exc
                        try:
                            fallback_strategy = dict(memory_strategy)
                            fallback_strategy["offload_strategy"] = "model_cpu_offload"
                            fallback_strategy["cuda_mode"] = "offload"
                            if selected_backend == "diffusers_gguf" and gguf_transformer_path:
                                try:
                                    del pipe
                                except Exception:
                                    pass
                                try:
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                except Exception:
                                    pass
                                pipe = _build_gguf_pipe(use_device_map_cuda=False)
                            else:
                                pipe = _load_pipeline(low_mem=True)
                            _enable_cuda_offload(pipe, fallback_strategy)
                            memory_strategy = fallback_strategy
                            placed = True
                        except Exception as offload_exc:
                            raise RuntimeError(
                                "CUDA out of memory while loading the Wan pipeline, and CPU offload fallback failed. "
                                f"GPU error: {exc}; offload error: {offload_exc}"
                            ) from offload_exc
                    else:
                        raise
        elif device == "cpu":
            pipe = pipe.to("cpu")

        _apply_pipeline_memory_optimizations(pipe, memory_strategy)
        _PIPELINE_MEMORY_STRATEGY = memory_strategy
        _PIPELINE = pipe
        _PIPELINE_KEY = key
        return pipe


def _unload_pipeline() -> None:
    global _PIPELINE, _PIPELINE_KEY, _PIPELINE_MEMORY_STRATEGY
    with _PIPELINE_LOCK:
        pipe = _PIPELINE
        _PIPELINE = None
        _PIPELINE_KEY = None
        _PIPELINE_MEMORY_STRATEGY = None
    if pipe is None:
        return
    try:
        import torch  # type: ignore

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _write_video_outputs(job: dict[str, Any], frames: list[Any], config: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image  # type: ignore

    job_id = str(job.get("job_id") or uuid.uuid4().hex)
    params = job.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    root = _resolve_output_dir(config)
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{job_id}"
    run_dir = root / run_id
    run_dir.mkdir(exist_ok=True, parents=True)

    poster_path = run_dir / "poster.jpg"
    video_path = run_dir / "video.mp4"
    meta_path = run_dir / "meta.json"

    frames = _normalize_frames_to_pil(frames)

    first = frames[0]
    if isinstance(first, Image.Image):
        first.save(poster_path, quality=92)
    else:
        raise RuntimeError("Pipeline frames are not PIL images; unsupported output format.")

    export_error = None
    try:
        from diffusers.utils import export_to_video  # type: ignore

        export_to_video(frames, str(video_path), fps=int(params.get("fps") or 16))
    except Exception as exc:
        export_error = str(exc)
        # Preserve something useful for inspection if video export failed.
        frames_dir = run_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        for idx, frame in enumerate(frames[:120]):
            frame.save(frames_dir / f"{idx:04d}.png")

    meta = {
        "plugin": "duckmotion",
        "job_id": job_id,
        "run_id": run_id,
        "created_at": _now(),
        "input": job.get("input") or {},
        "params": params,
        "runtime_profile": job.get("runtime_profile") or {},
        "frame_count": len(frames),
        "export": {
            "video_path": str(video_path) if video_path.exists() else None,
            "poster_path": str(poster_path),
            "export_error": export_error,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    item = _gallery_item_from_run(run_dir)
    if item is None:
        raise RuntimeError("Failed to build gallery item for generated output.")
    return item


# ---------------------------------------------------------------------------
# Gallery scanning / file URL helpers
# ---------------------------------------------------------------------------


def _scan_gallery(root: Path, limit: int = 100) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    items: list[dict[str, Any]] = []
    for run in runs:
        item = _gallery_item_from_run(run)
        if item is not None:
            items.append(item)
            if len(items) >= limit:
                break
    return items


def _gallery_item_from_run(run_dir: Path) -> dict[str, Any] | None:
    if not run_dir.exists() or not run_dir.is_dir():
        return None
    meta_path = run_dir / "meta.json"
    poster_path = run_dir / "poster.jpg"

    video_path = None
    for candidate in sorted(run_dir.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES:
            video_path = candidate
            break

    stat = run_dir.stat()
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                meta = raw
        except Exception:
            meta = {}

    return {
        "run_id": run_dir.name,
        "mtime": float(stat.st_mtime),
        "video": _artifact_descriptor(video_path) if video_path else None,
        "poster": _artifact_descriptor(poster_path) if poster_path.exists() else None,
        "meta": meta,
    }


def _artifact_descriptor(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "url": _artifact_url(path),
        "web_path": _safe_web_path_for_any(path),
    }


def _artifact_url(path: Path) -> str:
    base_resolved = BASE.resolve()
    try:
        path.resolve().relative_to(base_resolved)
        return f"/{to_web_path(path)}"
    except Exception:
        run_id = path.parent.name
        return f"./gallery/file/{run_id}/{path.name}"


def _safe_web_path_for_any(path: Path) -> str | None:
    try:
        path.resolve().relative_to(BASE.resolve())
        return f"/{to_web_path(path)}"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Input image staging + WebbDuck output discovery
# ---------------------------------------------------------------------------


def _resolve_input_image(path_str: str) -> Path:
    raw = str(path_str or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="image_path is required.")
    target = resolve_web_path(raw) if ("outputs/" in raw or raw.startswith("/outputs/")) else Path(raw)
    target = target.expanduser()
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Input image not found.")
    if target.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Input file must be a PNG/JPG/WEBP image.")
    return target


def _staging_item(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "web_path": f"/{to_web_path(path)}",
        "size_bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
    }


def _recent_webbduck_images(limit: int = 24) -> list[dict[str, Any]]:
    try:
        runs = [p for p in BASE.iterdir() if p.is_dir() and p.name != STAGING_DIR.name and p.name != DEFAULT_OUTPUT_DIR.name]
    except Exception:
        return []
    runs.sort(key=lambda p: p.name, reverse=True)

    out: list[dict[str, Any]] = []
    for run in runs:
        for img in sorted(run.iterdir(), key=lambda p: p.name):
            if not img.is_file() or img.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            if img.name.endswith("_upscaled.png"):
                continue
            out.append(
                {
                    "run": run.name,
                    "name": img.name,
                    "web_path": f"/{to_web_path(img)}",
                    "path": str(img),
                    "mtime": _safe_mtime(img),
                }
            )
            if len(out) >= limit:
                return out
    return out


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _normalize_generate_params(payload: GeneratePayload, config: dict[str, Any]) -> dict[str, Any]:
    width = _clamp_int(payload.width, default=int(config.get("default_width") or 832), lo=256, hi=1920)
    height = _clamp_int(payload.height, default=int(config.get("default_height") or 480), lo=256, hi=1920)
    width, height = _snap_dimensions(width, height)
    return {
        "prompt": str(payload.prompt or "").strip(),
        "negative_prompt": str(payload.negative_prompt or "").strip(),
        "width": width,
        "height": height,
        "num_frames": _clamp_int(payload.num_frames, default=int(config.get("default_frames") or 81), lo=8, hi=241),
        "fps": _clamp_int(payload.fps, default=int(config.get("default_fps") or 16), lo=1, hi=60),
        "num_inference_steps": _clamp_int(
            payload.num_inference_steps,
            default=int(config.get("default_steps") or 30),
            lo=1,
            hi=120,
        ),
        "guidance_scale": _clamp_float(
            payload.guidance_scale,
            default=float(config.get("default_guidance_scale") or 5.0),
            lo=0.0,
            hi=20.0,
        ),
        "seed": int(payload.seed) if payload.seed is not None else None,
    }


def _snap_dimensions(width: int, height: int) -> tuple[int, int]:
    # Wan/diffusers video pipelines generally expect multiples of 16.
    w = max(256, (int(width) // 16) * 16)
    h = max(256, (int(height) // 16) * 16)
    return (w, h)


def _normalize_path_string(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        raw = raw[1:-1].strip()
    return raw


def _normalize_str(value: Any) -> str:
    return str(value or "").strip()


def _clamp_int(value: Any, *, default: int, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except Exception:
        return int(default)
    return max(int(lo), min(int(hi), n))


def _clamp_float(value: Any, *, default: float, lo: float, hi: float) -> float:
    try:
        n = float(value)
    except Exception:
        return float(default)
    return max(float(lo), min(float(hi), n))


def _safe_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _safe_stem(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    return clean.strip("._-")[:120]


def _resolve_cuda_mode(config: dict[str, Any]) -> str:
    env_mode = _normalize_str(os.getenv("DUCKMOTION_CUDA_MODE") or "")
    cfg_mode = _normalize_str(config.get("cuda_mode") or "")
    mode = (env_mode or cfg_mode or "offload").lower()
    if mode not in {"offload", "full"}:
        return "offload"
    return mode


def _has_explicit_cuda_mode(config: dict[str, Any]) -> bool:
    return bool(_normalize_str(os.getenv("DUCKMOTION_CUDA_MODE") or "") or _normalize_str(config.get("cuda_mode") or ""))


def _resolve_effective_cuda_mode(
    config: dict[str, Any],
    runtime_profile: dict[str, Any],
    selected_backend: str,
) -> str:
    explicit = _resolve_cuda_mode(config)
    if _has_explicit_cuda_mode(config):
        return explicit

    device = str(runtime_profile.get("device") or "cpu").lower()
    raw_vram = runtime_profile.get("total_vram_gb")
    total_vram_gb = float(raw_vram) if isinstance(raw_vram, (int, float)) else None
    if device != "cuda":
        return "offload"

    if selected_backend == "diffusers_gguf":
        # Quantized transformers can tolerate earlier GPU placement, but 16 GB-class
        # Windows systems still tend to OOM during base-pipeline materialization.
        if total_vram_gb is not None and total_vram_gb >= 24.0:
            return "full"
        return "offload"

    if total_vram_gb is not None and total_vram_gb >= 24.0:
        return "full"
    return "offload"


def _resolve_effective_offload_strategy(
    memory_strategy: dict[str, Any],
    runtime_profile: dict[str, Any],
    selected_backend: str,
) -> str:
    strategy = str(memory_strategy.get("offload_strategy") or "none")
    if strategy == "none":
        return strategy

    device = str(runtime_profile.get("device") or "cpu").lower()
    raw_vram = runtime_profile.get("total_vram_gb")
    total_vram_gb = float(raw_vram) if isinstance(raw_vram, (int, float)) else None
    if device != "cuda":
        return strategy

    if selected_backend == "diffusers_gguf":
        if total_vram_gb is not None and total_vram_gb <= 12.0:
            return "sequential_cpu_offload"
        return "group_offload"

    return strategy


def _get_webbduck_cleanup_hooks() -> tuple[Any | None, Any | None]:
    pipeline_manager = None
    unload_captioners = None
    try:
        from webbduck.core.pipeline import pipeline_manager as imported_pipeline_manager  # type: ignore

        pipeline_manager = imported_pipeline_manager
    except Exception:
        pipeline_manager = None
    try:
        from webbduck.core.captioner import unload_captioners as imported_unload_captioners  # type: ignore

        unload_captioners = imported_unload_captioners
    except Exception:
        unload_captioners = None
    return pipeline_manager, unload_captioners


def _cleanup_torch_memory() -> None:
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            use_ipc_collect = str(os.getenv("WEBBDUCK_USE_IPC_COLLECT", "")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if use_ipc_collect and hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _prepare_runtime_for_wan(runtime_profile: dict[str, Any]) -> None:
    device = str(runtime_profile.get("device") or "cpu").lower()
    t0 = time.perf_counter()
    pipeline_unloaded = False
    captioners_unloaded = False
    pipeline_manager, unload_captioners = _get_webbduck_cleanup_hooks()

    if pipeline_manager is not None and hasattr(pipeline_manager, "unload_all"):
        try:
            pipeline_manager.unload_all()
            pipeline_unloaded = True
        except Exception:
            _LOG.exception("DuckMotion failed to unload WebbDuck pipelines before Wan load.")

    if callable(unload_captioners):
        try:
            unload_captioners()
            captioners_unloaded = True
        except Exception:
            _LOG.exception("DuckMotion failed to unload WebbDuck captioners before Wan load.")

    _cleanup_torch_memory()
    elapsed = time.perf_counter() - t0
    _LOG.info(
        "DuckMotion preflight cleanup complete (device=%s, pipelines_unloaded=%s, captioners_unloaded=%s, %.2fs).",
        device,
        pipeline_unloaded,
        captioners_unloaded,
        elapsed,
    )


def _normalize_memory_policy(value: Any) -> str:
    raw = _normalize_str(value or "").lower()
    if not raw:
        return "auto"
    aliases = {
        "none": "off",
        "disabled": "off",
        "false": "off",
        "default": "balanced",
        "safe": "balanced",
        "on": "balanced",
        "low-vram": "aggressive",
        "low_vram": "aggressive",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"auto", "off", "balanced", "aggressive"}:
        return "auto"
    return normalized


def _normalize_safety_mode(value: Any) -> str:
    raw = _normalize_str(value or "").lower()
    if not raw:
        return "block"
    aliases = {
        "strict": "block",
        "default": "block",
        "warn-only": "warn",
        "warn_only": "warn",
        "disabled": "off",
        "false": "off",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"block", "warn", "off"}:
        return "block"
    return normalized


def _normalize_runtime_backend(value: Any) -> str:
    raw = _normalize_str(value or "").lower()
    if not raw:
        return "auto"
    aliases = {
        "default": "auto",
        "gguf": "diffusers_gguf",
        "diffusers-gguf": "diffusers_gguf",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"auto", "diffusers", "diffusers_gguf"}:
        return "auto"
    return normalized


def _is_quantized_transformer_path(value: str) -> bool:
    raw = _normalize_path_string(value)
    return raw.lower().endswith((".gguf", ".safetensors"))


def _find_matching_gguf_candidate(
    gguf_path: str,
    gguf_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    target = _normalize_path_string(gguf_path).lower()
    if not target:
        return None
    candidates = gguf_candidates if isinstance(gguf_candidates, list) else []
    for candidate in candidates:
        paths = candidate.get("paths") or {}
        if not isinstance(paths, dict):
            continue
        for key in ("H", "single", "L"):
            candidate_path = _normalize_path_string(paths.get(key) or "").lower()
            if candidate_path and candidate_path == target:
                return candidate
    return None


def _resolve_gguf_transformer_path(
    config: dict[str, Any],
    discovered_gguf_items: list[dict[str, Any]] | None = None,
    gguf_candidates: list[dict[str, Any]] | None = None,
) -> str:
    configured = _normalize_path_string(config.get("gguf_transformer_path") or "")
    if configured:
        return configured
    model_source = _normalize_path_string(config.get("model_id_or_path") or "")
    if model_source.startswith("gguf:"):
        model_source = model_source[5:]
    if _is_quantized_transformer_path(model_source):
        return model_source
    requested = _normalize_runtime_backend(config.get("runtime_backend") or os.getenv("DUCKMOTION_RUNTIME_BACKEND") or "auto")
    if requested != "diffusers_gguf":
        return ""
    candidates = gguf_candidates if isinstance(gguf_candidates, list) else []
    for candidate in candidates:
        paths = candidate.get("paths") or {}
        if not isinstance(paths, dict):
            continue
        preferred = _normalize_path_string(paths.get("H") or paths.get("single") or "")
        if preferred:
            return preferred
    items = discovered_gguf_items if isinstance(discovered_gguf_items, list) else []
    if items:
        preferred = items[0]
        return _normalize_path_string(preferred.get("path") or "")
    return ""


def _plan_runtime_backend(
    config: dict[str, Any],
    runtime_profile: dict[str, Any] | None,
    *,
    diffusers_status: dict[str, Any] | None = None,
    discovered_gguf_items: list[dict[str, Any]] | None = None,
    gguf_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    requested = _normalize_runtime_backend(config.get("runtime_backend") or os.getenv("DUCKMOTION_RUNTIME_BACKEND") or "auto")
    gguf_path = _resolve_gguf_transformer_path(config, discovered_gguf_items, gguf_candidates)
    candidate_rows = gguf_candidates if isinstance(gguf_candidates, list) else []
    selected_candidate = _find_matching_gguf_candidate(gguf_path, candidate_rows)
    gguf_available = bool(gguf_path)
    gguf_option_available = bool(candidate_rows or (discovered_gguf_items if isinstance(discovered_gguf_items, list) else []))
    diffusers_ready = bool((diffusers_status or {}).get("ready")) if isinstance(diffusers_status, dict) else True
    gguf_ready = bool((diffusers_status or {}).get("gguf_ready")) if isinstance(diffusers_status, dict) else False
    total_vram_gb = None
    device = "unknown"
    if isinstance(runtime_profile, dict):
        device = str(runtime_profile.get("device") or "unknown").lower()
        raw_vram = runtime_profile.get("total_vram_gb")
        if isinstance(raw_vram, (int, float)):
            total_vram_gb = float(raw_vram)

    selected = requested
    reason = "user-selected backend"
    if requested == "auto":
        selected = "diffusers"
        reason = "default diffusers runtime"
        if gguf_available and gguf_ready and device == "cuda" and isinstance(total_vram_gb, (int, float)) and total_vram_gb <= 16.0:
            selected = "diffusers_gguf"
            reason = "GGUF transformer available and lower-VRAM CUDA runtime detected"
    elif requested == "diffusers_gguf" and not (gguf_available and gguf_ready):
        selected = "diffusers"
        reason = "requested GGUF backend unavailable; falling back to diffusers"

    available_backends = ["diffusers"]
    if gguf_ready and gguf_option_available:
        available_backends.append("diffusers_gguf")

    # For GGUF backends, ensure a valid base model path is available.
    # If no local diffusers folder was found alongside the GGUF file, fall back to the HF repo ID.
    gguf_base_model = None
    if selected_candidate and isinstance(selected_candidate, dict):
        gguf_base_model = selected_candidate.get("base_model_path")
    if not gguf_base_model:
        model_source = _normalize_path_string(config.get("model_id_or_path") or "")
        if model_source.startswith("gguf:"):
            model_source = model_source[5:]
        if model_source and not _is_quantized_transformer_path(model_source):
            gguf_base_model = model_source
    if selected == "diffusers_gguf" and not gguf_base_model:
        gguf_base_model = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

    return {
        "requested_backend": requested,
        "selected_backend": selected,
        "available_backends": available_backends,
        "gguf_transformer_path": gguf_path or None,
        "gguf_base_model_path": gguf_base_model,
        "gguf_available": gguf_available,
        "gguf_candidate": selected_candidate,
        "gguf_pair_complete": bool((selected_candidate or {}).get("complete")) if isinstance(selected_candidate, dict) else False,
        "diffusers_ready": diffusers_ready,
        "gguf_ready": gguf_ready,
        "reason": reason,
    }


def _get_current_setup_compatibility(config: dict[str, Any], runtime_profile: dict[str, Any]) -> dict[str, Any]:
    default_params = {
        "width": _clamp_int(config.get("default_width"), default=832, lo=256, hi=1920),
        "height": _clamp_int(config.get("default_height"), default=480, lo=256, hi=1920),
        "num_frames": _clamp_int(config.get("default_frames"), default=81, lo=8, hi=241),
        "num_inference_steps": _clamp_int(config.get("default_steps"), default=30, lo=1, hi=120),
    }
    discovered = _discover_local_models(config)
    diffusers_status = _probe_diffusers_support()
    runtime_plan = _plan_runtime_backend(
        config,
        runtime_profile,
        diffusers_status=diffusers_status,
        discovered_gguf_items=discovered.get("gguf_items"),
        gguf_candidates=discovered.get("gguf_candidates"),
    )
    strict_config = dict(config)
    strict_config["safety_mode"] = "block"
    safety = _evaluate_job_safety(default_params, strict_config, runtime_profile)
    compatible = bool(safety.get("ok"))
    reason = "Compatible with current setup."
    if not compatible:
        if runtime_plan.get("selected_backend") == "diffusers_gguf":
            reason = "Compatible via GGUF backend (quantized transformer)."
            compatible = True
        else:
            reason = str(safety.get("message") or "Incompatible with current setup.")
    return {
        "compatible": compatible,
        "reason": reason,
        "runtime_plan": runtime_plan,
        "safety": safety,
    }


def _get_host_memory_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"os": os.name}
    if os.name != "nt":
        return snapshot

    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            snapshot.update(
                {
                    "memory_load_percent": int(stat.dwMemoryLoad),
                    "total_phys_gb": round(stat.ullTotalPhys / 1024**3, 2),
                    "avail_phys_gb": round(stat.ullAvailPhys / 1024**3, 2),
                    "total_pagefile_gb": round(stat.ullTotalPageFile / 1024**3, 2),
                    "avail_pagefile_gb": round(stat.ullAvailPageFile / 1024**3, 2),
                }
            )

        class PERFORMANCE_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("CommitTotal", ctypes.c_size_t),
                ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t),
                ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t),
                ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t),
                ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t),
                ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_ulong),
                ("ProcessCount", ctypes.c_ulong),
                ("ThreadCount", ctypes.c_ulong),
            ]

        perf = PERFORMANCE_INFORMATION()
        perf.cb = ctypes.sizeof(PERFORMANCE_INFORMATION)
        if ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(perf), perf.cb):
            page_size = int(perf.PageSize)
            commit_limit_bytes = int(perf.CommitLimit) * page_size
            commit_total_bytes = int(perf.CommitTotal) * page_size
            physical_total_bytes = int(perf.PhysicalTotal) * page_size
            estimated_pagefile_bytes = max(0, commit_limit_bytes - physical_total_bytes)
            snapshot.update(
                {
                    "commit_limit_gb": round(commit_limit_bytes / 1024**3, 2),
                    "commit_total_gb": round(commit_total_bytes / 1024**3, 2),
                    "commit_headroom_gb": round(max(0, commit_limit_bytes - commit_total_bytes) / 1024**3, 2),
                    "configured_pagefile_gb": round(estimated_pagefile_bytes / 1024**3, 2),
                }
            )
    except Exception as exc:
        snapshot["error"] = str(exc)
    return snapshot


def _estimate_job_pressure(params: dict[str, Any]) -> dict[str, Any]:
    width = int(params.get("width") or 832)
    height = int(params.get("height") or 480)
    frames = int(params.get("num_frames") or 81)
    steps = int(params.get("num_inference_steps") or 30)
    megapixels = (width * height) / 1_000_000.0
    pressure_score = round(megapixels * max(1.0, frames / 49.0) * max(1.0, steps / 20.0), 2)
    return {
        "width": width,
        "height": height,
        "num_frames": frames,
        "num_inference_steps": steps,
        "megapixels": round(megapixels, 3),
        "pressure_score": pressure_score,
    }


def _safer_generation_recommendations() -> list[str]:
    return [
        "Try 576x320, 49 frames, and 20 steps.",
        "Use DUCKMOTION_CUDA_MODE=offload instead of full on lower-VRAM systems.",
        "Use DUCKMOTION_MEMORY_POLICY=aggressive to prefer the lowest-memory runtime path.",
        "If the system is still unstable, reduce resolution or frames further before retrying.",
    ]


def _evaluate_job_safety(params: dict[str, Any], config: dict[str, Any], runtime_profile: dict[str, Any]) -> dict[str, Any]:
    safety_mode = _normalize_safety_mode(os.getenv("DUCKMOTION_SAFETY_MODE") or config.get("safety_mode") or "block")
    memory_strategy = _plan_memory_strategy(config, runtime_profile)
    host_memory = _get_host_memory_snapshot()
    pressure = _estimate_job_pressure(params)
    device = str(runtime_profile.get("device") or "cpu").lower()
    total_vram_gb = runtime_profile.get("total_vram_gb")
    avail_phys_gb = host_memory.get("avail_phys_gb")
    avail_pagefile_gb = host_memory.get("avail_pagefile_gb")
    configured_pagefile_gb = host_memory.get("configured_pagefile_gb")
    model_source = _normalize_str(config.get("model_id_or_path") or "")
    is_wan_a14b = "wan2.2-i2v-a14b" in model_source.lower()
    configured_gguf_path = _resolve_gguf_transformer_path(config)
    requested_backend = _normalize_runtime_backend(config.get("runtime_backend") or os.getenv("DUCKMOTION_RUNTIME_BACKEND") or "auto")
    using_gguf_backend = requested_backend == "diffusers_gguf" and bool(configured_gguf_path)
    plan: dict[str, Any] = {}
    reasons: list[str] = []

    if os.name == "nt" and device == "cuda":
        if is_wan_a14b and isinstance(total_vram_gb, (int, float)) and float(total_vram_gb) <= 16.0:
            # A14B safety checks only apply to full diffusers loads — GGUF loads a quantized
            # transformer instead and does not trigger the same memory pressure.
            # Skip A14B block when a GGUF transformer is available and selected.
            discovered = _discover_local_models(config)
            plan = _plan_runtime_backend(
                config,
                runtime_profile,
                diffusers_status=None,
                discovered_gguf_items=discovered.get("gguf_items"),
                gguf_candidates=discovered.get("gguf_candidates"),
            )
            using_gguf_backend = using_gguf_backend or plan.get("selected_backend") == "diffusers_gguf"
            if not using_gguf_backend:
                if isinstance(configured_pagefile_gb, (int, float)) and float(configured_pagefile_gb) < 32.0:
                    reasons.append("Wan2.2 A14B diffusers loading is unsafe on this Windows system with the current configured page file and VRAM tier.")
                elif configured_pagefile_gb is None and pressure["pressure_score"] >= 0.75:
                    reasons.append("Wan2.2 A14B diffusers loading is high risk on this Windows VRAM tier and page-file capacity could not be verified.")
        using_quantized_backend = using_gguf_backend or plan.get("selected_backend") == "diffusers_fp8"
        if not using_quantized_backend:
            if isinstance(configured_pagefile_gb, (int, float)) and float(configured_pagefile_gb) < 16.0 and pressure["pressure_score"] >= 0.75:
                reasons.append("Configured Windows page file is too small for this Wan workload.")
            if memory_strategy.get("cuda_mode") == "full" and isinstance(total_vram_gb, (int, float)) and float(total_vram_gb) <= 16.0:
                if pressure["pressure_score"] >= 0.9:
                    reasons.append("Full CUDA mode is high risk on Windows for this GPU memory tier and job size.")
            if isinstance(avail_pagefile_gb, (int, float)) and float(avail_pagefile_gb) < 12.0 and pressure["pressure_score"] >= 0.9:
                reasons.append("Available Windows page file is low for this Wan workload.")
            if isinstance(avail_phys_gb, (int, float)) and float(avail_phys_gb) < 8.0 and pressure["pressure_score"] >= 0.9:
                reasons.append("Available system RAM is low for this Wan workload.")
            if pressure["width"] >= 832 and pressure["height"] >= 480 and pressure["num_frames"] >= 81 and pressure["num_inference_steps"] >= 30:
                if isinstance(total_vram_gb, (int, float)) and float(total_vram_gb) <= 16.0:
                    reasons.append("The current DuckMotion default-size Wan job is too aggressive for this detected VRAM tier on Windows.")
            if memory_strategy.get("effective_policy") == "aggressive" and pressure["pressure_score"] >= 1.5:
                reasons.append("Even the aggressive low-memory strategy predicts elevated crash risk for this request.")

    blocked = bool(reasons) and safety_mode == "block"
    result = {
        "ok": not blocked,
        "blocked": blocked,
        "safety_mode": safety_mode,
        "risk_level": "high" if reasons else "low",
        "reasons": reasons,
        "recommendations": _safer_generation_recommendations() if reasons else [],
        "memory_strategy": memory_strategy,
        "host_memory": host_memory,
        "pressure": pressure,
    }
    if reasons:
        prefix = "Blocked unsafe DuckMotion Wan job." if blocked else "DuckMotion safety warning for risky Wan job."
        result["message"] = f"{prefix} {' '.join(reasons)}"
    else:
        result["message"] = "DuckMotion preflight safety check passed."
    return result


def _plan_memory_strategy(config: dict[str, Any], runtime_profile: dict[str, Any]) -> dict[str, Any]:
    requested = _normalize_memory_policy(os.getenv("DUCKMOTION_MEMORY_POLICY") or config.get("memory_policy") or "auto")
    device = str(runtime_profile.get("device") or "cpu").lower()
    dtype = str(runtime_profile.get("dtype") or "float32").lower()
    cuda_mode = _resolve_cuda_mode(config)
    raw_vram = runtime_profile.get("total_vram_gb")
    total_vram_gb = float(raw_vram) if isinstance(raw_vram, (int, float)) else None

    effective = requested
    notes: list[str] = []
    if requested == "auto":
        if device != "cuda":
            effective = "off"
            notes.append("auto policy disabled VRAM reductions because runtime device is not CUDA")
        elif total_vram_gb is not None and total_vram_gb <= 12.0:
            effective = "aggressive"
            notes.append("auto policy selected aggressive mode for low-VRAM CUDA runtime")
        elif cuda_mode != "full":
            effective = "balanced"
            notes.append("auto policy selected balanced mode for CUDA offload runtime")
        elif total_vram_gb is not None and total_vram_gb <= 20.0:
            effective = "balanced"
            notes.append("auto policy selected balanced mode for mid-VRAM full-CUDA runtime")
        else:
            effective = "off"
            notes.append("auto policy left extra VRAM reductions off for higher-VRAM full-CUDA runtime")

    attention_mode = None
    if effective == "balanced":
        attention_mode = "auto"
    elif effective == "aggressive":
        attention_mode = "max"

    offload_strategy = "none"
    if device == "cuda" and cuda_mode != "full":
        offload_strategy = "sequential_cpu_offload" if effective == "aggressive" else "model_cpu_offload"

    return {
        "requested_policy": requested,
        "effective_policy": effective,
        "device": device,
        "dtype": dtype,
        "cuda_mode": cuda_mode,
        "total_vram_gb": total_vram_gb,
        "offload_strategy": offload_strategy,
        "attention_slicing": attention_mode,
        "vae_slicing": effective in {"balanced", "aggressive"},
        "vae_tiling": effective == "aggressive",
        "notes": notes,
    }


def _enable_cuda_offload(pipe: Any, memory_strategy: dict[str, Any]) -> None:
    strategy = str(memory_strategy.get("offload_strategy") or "model_cpu_offload")
    method_map = {
        "model_cpu_offload": "enable_model_cpu_offload",
        "sequential_cpu_offload": "enable_sequential_cpu_offload",
        "group_offload": "enable_model_cpu_offload",
    }

    errors: list[str] = []
    if strategy == "group_offload":
        method = getattr(pipe, "enable_group_offload", None)
        if callable(method):
            try:
                import torch  # type: ignore

                method(
                    onload_device=torch.device("cuda"),
                    offload_device=torch.device("cpu"),
                    offload_type="block_level",
                    num_blocks_per_group=1,
                    low_cpu_mem_usage=True,
                )
                # Workaround: accelerate's block_level group_offload registers per-block hooks
                # but does not move non-block modules (patch_embedding, condition_embedder, etc.)
                # to the onload_device. Without this fix those modules stay on CPU while the
                # pipeline sends input tensors to CUDA, causing a silent device mismatch that
                # makes the forward pass fall back to CPU execution (13+ minute hang at step 0).
                transformer = getattr(pipe, "transformer", None)
                if transformer is not None and hasattr(transformer, "blocks"):
                    for child_name, child_mod in list(transformer.named_children()):
                        if child_name != "blocks":
                            try:
                                child_mod.to(torch.device("cuda"))
                            except Exception as move_exc:
                                _LOG.warning(
                                    "group_offload non-block fix: could not move transformer.%s to cuda: %s",
                                    child_name, move_exc,
                                )
                memory_strategy["offload_strategy"] = "group_offload"
                return
            except Exception as exc:
                errors.append(f"enable_group_offload: {exc}")
        else:
            errors.append("enable_group_offload unavailable")

    methods = [method_map.get(strategy, "enable_model_cpu_offload")]
    if methods[0] != "enable_model_cpu_offload":
        methods.append("enable_model_cpu_offload")

    for method_name in methods:
        method = getattr(pipe, method_name, None)
        if not callable(method):
            errors.append(f"{method_name} unavailable")
            continue
        try:
            method()
            memory_strategy["offload_strategy"] = (
                "sequential_cpu_offload" if method_name == "enable_sequential_cpu_offload" else "model_cpu_offload"
            )
            return
        except Exception as exc:
            errors.append(f"{method_name}: {exc}")

    detail = "; ".join(errors) if errors else "no compatible CPU offload method available"
    raise RuntimeError(detail)


def _apply_pipeline_memory_optimizations(pipe: Any, memory_strategy: dict[str, Any]) -> None:
    applied: list[str] = []
    skipped: list[str] = []

    attention_mode = memory_strategy.get("attention_slicing")
    if attention_mode:
        method = getattr(pipe, "enable_attention_slicing", None)
        if callable(method):
            try:
                method(attention_mode)
                applied.append(f"attention_slicing={attention_mode}")
            except TypeError:
                try:
                    method()
                    applied.append("attention_slicing")
                except Exception as exc:
                    skipped.append(f"attention_slicing ({exc})")
            except Exception as exc:
                skipped.append(f"attention_slicing ({exc})")
        else:
            skipped.append("attention_slicing unavailable")

    for attr_name, method_name in (
        ("vae_slicing", "enable_vae_slicing"),
        ("vae_tiling", "enable_vae_tiling"),
    ):
        if not memory_strategy.get(attr_name):
            continue
        method = getattr(pipe, method_name, None)
        if not callable(method):
            skipped.append(f"{method_name} unavailable")
            continue
        try:
            method()
            applied.append(method_name)
        except Exception as exc:
            skipped.append(f"{method_name} ({exc})")

    memory_strategy["applied_optimizations"] = applied
    memory_strategy["skipped_optimizations"] = skipped
    if applied:
        _LOG.info(
            "DuckMotion memory strategy requested=%s effective=%s offload=%s applied=%s",
            memory_strategy.get("requested_policy"),
            memory_strategy.get("effective_policy"),
            memory_strategy.get("offload_strategy"),
            ", ".join(applied),
        )
    if skipped:
        _LOG.info("DuckMotion memory optimizations skipped: %s", "; ".join(skipped))


def _allow_unsafe_load_fallbacks() -> bool:
    raw = _normalize_str(os.getenv("DUCKMOTION_ALLOW_UNSAFE_LOAD_FALLBACKS") or "")
    return raw.lower() in {"1", "true", "yes", "on"}


def _should_isolate_process() -> bool:
    raw = _normalize_str(os.getenv("DUCKMOTION_ISOLATE_PROCESS") or "")
    if raw:
        return raw.lower() in {"1", "true", "yes", "on"}
    # Default to isolated generation on Windows so plugin crashes cannot terminate WebbDuck.
    return os.name == "nt"


def _should_keep_pipeline_loaded(config: dict[str, Any]) -> bool:
    raw = _normalize_str(os.getenv("DUCKMOTION_KEEP_PIPELINE_LOADED") or config.get("keep_pipeline_loaded") or "")
    if raw:
        return raw.lower() in {"1", "true", "yes", "on"}
    # Default to keeping quantized Wan pipelines warm on Windows because the
    # fragile part of these runs is repeated base-pipeline reload, not steady-state inference.
    if os.name == "nt" and _normalize_path_string(config.get("gguf_transformer_path") or ""):
        return True
    return False


def _is_cuda_oom(exc: Exception) -> bool:
    text = str(exc or "").lower()
    if "out of memory" in text:
        return True
    if "cuda error: out of memory" in text:
        return True
    return False


def _is_windows_paging_file_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    if "paging file is too small" in text:
        return True
    if "os error 1455" in text:
        return True
    return isinstance(exc, MemoryError)


def _is_meta_tensor_move_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    if "cannot copy out of meta tensor" in text:
        return True
    if "module.to_empty()" in text:
        return True
    return False


def _is_missing_accelerate_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    if "accelerate" in text and "install" in text:
        return True
    if "requires accelerate" in text:
        return True
    if "accelerate is not installed" in text:
        return True
    return False


def _is_low_cpu_mem_usage_fp32_constraint_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    if "low_cpu_mem_usage" in text and "keep_in_fp32_modules" in text:
        return True
    return False


def _is_windows_access_violation_rc(code: int) -> bool:
    return int(code) in {3221225477, -1073741819}


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="DuckMotion backend helper entrypoint.")
    parser.add_argument("--duckmotion-child", action="store_true", help="Run a single isolated generation payload.")
    parser.add_argument("--payload", type=str, default="", help="Path to child payload JSON.")
    parser.add_argument("--result", type=str, default="", help="Path to child result JSON.")
    args = parser.parse_args()

    if not args.duckmotion_child:
        parser.print_help()
        return 2

    payload_path = Path(args.payload).expanduser() if args.payload else None
    result_path = Path(args.result).expanduser() if args.result else None
    if payload_path is None or result_path is None:
        return 2
    return _run_generation_child(payload_path, result_path)


if __name__ == "__main__":
    raise SystemExit(_main())
