"""DuckMotion (Wan2.2) web plugin backend for WebbDuck (Phase 1 scaffold).

Phase 1 focuses on:
- installation/readiness detection (GGUF pair + companion files)
- ComfyUI connectivity/configuration
- image staging (upload / copy from WebbDuck outputs)
- raw ComfyUI prompt bridge endpoints for advanced users

Guided Wan workflow submission (node patching/templates) is deferred to Phase 2.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from webbduck.server.storage import BASE, resolve_web_path, to_web_path

PLUGIN_ROOT = Path(__file__).resolve().parent
STATE_DIR = Path.home() / ".webbduck" / "plugin_state"
CONFIG_FILE = STATE_DIR / "duckmotion_config.json"
JOBS_FILE = STATE_DIR / "duckmotion_jobs.json"
STAGING_DIR = BASE / "duckmotion_staging"

REQUIRED_TEXT_ENCODER_NAMES = (
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
)
RECOMMENDED_VAE_NAMES = (
    "wan_2.1_vae.safetensors",
    "wan2.2_vae.safetensors",
)

GGUF_QUANT_RE = re.compile(r"(q\d(?:[_a-z0-9]+)?)", re.IGNORECASE)


class ConfigPayload(BaseModel):
    comfy_url: str | None = None
    models_dir: str | None = None
    comfy_models_dir: str | None = None


def get_router(_plugin_manifest: dict | None = None) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, Any]:
        config = _load_config()
        install = _detect_installation(config)
        comfy = _probe_comfy(config)
        return {
            "ok": True,
            "phase": 1,
            "config": _public_config(config),
            "installation": install,
            "comfy": comfy,
            "ready": {
                # "enabled" is the Phase 1 optional-plugin gate the user asked for.
                "plugin_enabled": bool(install["gguf_pair_found"]),
                "generation_prereqs": bool(
                    install["gguf_pair_found"] and install["companion_files_found"]
                ),
                "comfy_connected": bool(comfy.get("reachable")),
                "bridge_ready": bool(
                    install["gguf_pair_found"]
                    and install["companion_files_found"]
                    and comfy.get("reachable")
                ),
            },
            "notes": [
                "Phase 1 includes setup checks, staging, and raw ComfyUI prompt bridge.",
                "Guided Wan workflow submission (templated node patching) lands in Phase 2.",
            ],
        }

    @router.get("/config")
    def get_config() -> dict[str, Any]:
        return {"config": _public_config(_load_config())}

    @router.post("/config")
    def set_config(payload: ConfigPayload) -> dict[str, Any]:
        current = _load_config()
        next_config = {
            "comfy_url": _normalize_http_base(
                payload.comfy_url if payload.comfy_url is not None else current.get("comfy_url", "")
            )
            or "",
            "models_dir": _normalize_path_string(
                payload.models_dir if payload.models_dir is not None else current.get("models_dir", "")
            ),
            "comfy_models_dir": _normalize_path_string(
                payload.comfy_models_dir
                if payload.comfy_models_dir is not None
                else current.get("comfy_models_dir", "")
            ),
        }
        _save_config(next_config)
        return {"ok": True, "config": _public_config(next_config)}

    @router.get("/jobs")
    def list_jobs(limit: int = Query(default=20, ge=1, le=200), refresh: bool = Query(default=False)) -> dict:
        jobs = _load_jobs()
        jobs.sort(key=lambda j: float(j.get("submitted_at", 0.0)), reverse=True)
        rows = jobs[: int(limit)]
        if refresh and rows:
            config = _load_config()
            refreshed = []
            changed = False
            for row in rows:
                prompt_id = str(row.get("prompt_id", "")).strip()
                if not prompt_id:
                    refreshed.append(row)
                    continue
                enriched = _enrich_job_from_history(row, config)
                if enriched != row:
                    changed = True
                refreshed.append(enriched)
            if changed:
                _merge_jobs(refreshed)
            rows = refreshed
        return {"jobs": rows}

    @router.get("/jobs/{prompt_id}")
    def get_job(prompt_id: str, refresh: bool = Query(default=True)) -> dict[str, Any]:
        pid = str(prompt_id or "").strip()
        if not pid:
            raise HTTPException(status_code=400, detail="prompt_id is required.")

        jobs = _load_jobs()
        row = next((j for j in jobs if str(j.get("prompt_id", "")).strip() == pid), None)
        if row is None:
            row = {"prompt_id": pid, "submitted_at": 0.0, "status": "unknown"}

        if refresh:
            config = _load_config()
            row = _enrich_job_from_history(row, config)
            _merge_jobs([row])
        return {"job": row}

    @router.post("/jobs/clear")
    def clear_jobs() -> dict[str, Any]:
        _save_jobs([])
        return {"ok": True}

    @router.post("/staging/upload")
    async def staging_upload(image: UploadFile = File(...)) -> dict[str, Any]:
        filename = str(image.filename or "").strip()
        if not filename:
            raise HTTPException(status_code=400, detail="Missing filename.")

        suffix = Path(filename).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise HTTPException(status_code=400, detail="Unsupported image type.")

        STAGING_DIR.mkdir(exist_ok=True, parents=True)
        stem = _safe_stem(Path(filename).stem) or "wan-input"
        out_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{stem}{suffix}"
        out_path = STAGING_DIR / out_name

        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty upload.")
        out_path.write_bytes(data)

        return {
            "ok": True,
            "item": _staging_item(out_path),
        }

    @router.post("/staging/from-webbduck")
    def staging_from_webbduck(path: str = Form(...)) -> dict[str, Any]:
        target = resolve_web_path(path)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Source image not found.")
        suffix = target.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise HTTPException(status_code=400, detail="Source file is not a supported image.")

        STAGING_DIR.mkdir(exist_ok=True, parents=True)
        stem = _safe_stem(target.stem) or "webbduck"
        out_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{stem}{suffix}"
        out_path = STAGING_DIR / out_name
        out_path.write_bytes(target.read_bytes())
        return {"ok": True, "item": _staging_item(out_path)}

    @router.get("/staging")
    def list_staging(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
        STAGING_DIR.mkdir(exist_ok=True, parents=True)
        rows = [p for p in STAGING_DIR.iterdir() if p.is_file()]
        rows.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        items = [_staging_item(p) for p in rows[: int(limit)]]
        return {"items": items}

    @router.delete("/staging/{name}")
    def delete_staging(name: str) -> dict[str, Any]:
        value = Path(str(name or "")).name
        if not value:
            raise HTTPException(status_code=400, detail="Invalid name.")
        path = STAGING_DIR / value
        if not path.exists():
            raise HTTPException(status_code=404, detail="Staged file not found.")
        path.unlink()
        return {"ok": True, "deleted": value}

    @router.get("/comfy/system_stats")
    def comfy_system_stats() -> dict[str, Any]:
        config = _load_config()
        return _require_comfy_json(config, "GET", "/system_stats")

    @router.get("/comfy/queue")
    def comfy_queue() -> dict[str, Any]:
        config = _load_config()
        return _require_comfy_json(config, "GET", "/queue")

    @router.get("/comfy/history/{prompt_id}")
    def comfy_history(prompt_id: str) -> dict[str, Any]:
        config = _load_config()
        pid = str(prompt_id or "").strip()
        if not pid:
            raise HTTPException(status_code=400, detail="prompt_id is required.")
        return _require_comfy_json(config, "GET", f"/history/{urllib_parse.quote(pid)}")

    @router.post("/comfy/prompt")
    def comfy_submit_prompt(payload: dict[str, Any]) -> dict[str, Any]:
        config = _load_config()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object.")

        body = dict(payload)
        body.setdefault("client_id", f"duckmotion-{uuid.uuid4().hex}")
        result = _require_comfy_json(config, "POST", "/prompt", body)

        prompt_id = str(result.get("prompt_id", "")).strip()
        if prompt_id:
            job = {
                "prompt_id": prompt_id,
                "client_id": str(body.get("client_id") or ""),
                "submitted_at": time.time(),
                "status": "queued",
                "phase": "bridge",
                "workflow_keys": list(body.keys()),
            }
            _merge_jobs([job])
        return result

    @router.get("/comfy/view")
    def comfy_view(
        filename: str = Query(..., min_length=1),
        subfolder: str = Query(default=""),
        type: str = Query(default="output"),
        format: str | None = Query(default=None),
    ) -> Response:
        config = _load_config()
        base = _configured_comfy_base(config)
        if not base:
            raise HTTPException(status_code=400, detail="ComfyUI URL is not configured.")

        query = {
            "filename": filename,
            "subfolder": subfolder or "",
            "type": type or "output",
        }
        if format:
            query["format"] = format
        qs = urllib_parse.urlencode(query)
        raw = _comfy_request_raw(base, "GET", f"/view?{qs}", timeout_s=30)
        guessed = raw.get("content_type") or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        headers = {}
        if raw.get("content_disposition"):
            headers["Content-Disposition"] = str(raw["content_disposition"])
        return Response(content=raw.get("body", b""), media_type=guessed, headers=headers)

    @router.get("/webbduck/recent-images")
    def recent_webbduck_images(limit: int = Query(default=24, ge=1, le=200)) -> dict[str, Any]:
        return {"items": _recent_webbduck_images(limit=int(limit))}

    return router


def _default_config() -> dict[str, str]:
    return {
        "comfy_url": _normalize_http_base(os.getenv("DUCKMOTION_COMFY_URL")) or "",
        "models_dir": _normalize_path_string(
            os.getenv("DUCKMOTION_MODELS_DIR") or ""
        ),
        "comfy_models_dir": _normalize_path_string(
            os.getenv("DUCKMOTION_COMFY_MODELS_DIR") or ""
        ),
    }


def _load_config() -> dict[str, str]:
    config = _default_config()
    if CONFIG_FILE.exists():
        try:
            raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                config.update(
                    {
                        "comfy_url": _normalize_http_base(raw.get("comfy_url")) or "",
                        "models_dir": _normalize_path_string(raw.get("models_dir") or config["models_dir"]),
                        "comfy_models_dir": _normalize_path_string(raw.get("comfy_models_dir") or ""),
                    }
                )
        except Exception:
            pass
    return config


def _save_config(config: dict[str, str]) -> None:
    STATE_DIR.mkdir(exist_ok=True, parents=True)
    payload = {
        "comfy_url": _normalize_http_base(config.get("comfy_url")) or "",
        "models_dir": _normalize_path_string(config.get("models_dir") or ""),
        "comfy_models_dir": _normalize_path_string(config.get("comfy_models_dir") or ""),
    }
    CONFIG_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _public_config(config: dict[str, str]) -> dict[str, str]:
    return {
        "comfy_url": str(config.get("comfy_url") or ""),
        "models_dir": str(config.get("models_dir") or ""),
        "comfy_models_dir": str(config.get("comfy_models_dir") or ""),
    }


def _normalize_path_string(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        raw = raw[1:-1].strip()
    return raw


def _normalize_http_base(value: str | None) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urllib_parse.urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    base = urllib_parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))
    return base.rstrip("/")


def _configured_comfy_base(config: dict[str, str]) -> str | None:
    return _normalize_http_base(config.get("comfy_url"))


def _probe_comfy(config: dict[str, str]) -> dict[str, Any]:
    base = _configured_comfy_base(config)
    if not base:
        return {
            "configured": False,
            "reachable": False,
            "base_url": "",
            "error": "ComfyUI URL not configured.",
        }

    try:
        stats = _comfy_request_json(base, "GET", "/system_stats", timeout_s=4)
        devices = []
        if isinstance(stats, dict):
            sys_info = stats.get("system") or stats.get("devices")
            if isinstance(sys_info, list):
                devices = [str(item.get("name") or item) for item in sys_info[:4] if isinstance(item, dict)]
        return {
            "configured": True,
            "reachable": True,
            "base_url": base,
            "error": None,
            "system_stats": stats,
            "devices": devices,
        }
    except Exception as exc:
        return {
            "configured": True,
            "reachable": False,
            "base_url": base,
            "error": str(exc),
        }


def _detect_installation(config: dict[str, str]) -> dict[str, Any]:
    models_dir = Path(config.get("models_dir") or "")
    comfy_models_dir = Path(config.get("comfy_models_dir") or "") if (config.get("comfy_models_dir") or "").strip() else None

    roots: list[Path] = []
    if str(models_dir).strip():
        roots.append(models_dir.expanduser())
    if comfy_models_dir is not None and str(comfy_models_dir).strip():
        roots.append(comfy_models_dir.expanduser())

    gguf_files: list[Path] = []
    companion_files: dict[str, list[str]] = {"text_encoders": [], "vaes": []}
    scan_errors: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                name_l = p.name.lower()
                if p.suffix.lower() == ".gguf":
                    gguf_files.append(p)
                elif p.suffix.lower() == ".safetensors":
                    if p.name in REQUIRED_TEXT_ENCODER_NAMES and str(p) not in companion_files["text_encoders"]:
                        companion_files["text_encoders"].append(str(p))
                    if p.name in RECOMMENDED_VAE_NAMES and str(p) not in companion_files["vaes"]:
                        companion_files["vaes"].append(str(p))
        except Exception as exc:
            scan_errors.append(f"{root}: {exc}")

    i2v_ggufs = []
    for path in gguf_files:
        meta = _classify_gguf(path)
        if not meta:
            continue
        i2v_ggufs.append(meta)

    pairs = _build_pairs(i2v_ggufs)
    pair_found = bool(pairs)
    best_pair = pairs[0] if pairs else None
    companion_ok = bool(companion_files["text_encoders"]) and bool(companion_files["vaes"])

    missing: list[str] = []
    if not pair_found:
        missing.append("Wan2.2 I2V GGUF pair (high + low)")
    if not companion_files["text_encoders"]:
        missing.append(REQUIRED_TEXT_ENCODER_NAMES[0])
    if not companion_files["vaes"]:
        missing.append("Wan VAE (wan_2.1_vae.safetensors or wan2.2_vae.safetensors)")

    return {
        "models_dir": str(models_dir.expanduser()) if str(models_dir).strip() else "",
        "comfy_models_dir": str(comfy_models_dir.expanduser()) if comfy_models_dir else "",
        "models_dir_exists": bool(str(models_dir).strip()) and models_dir.expanduser().exists(),
        "comfy_models_dir_exists": bool(comfy_models_dir and comfy_models_dir.expanduser().exists()),
        "scan_roots": [str(p) for p in roots],
        "scan_errors": scan_errors,
        "gguf_pair_found": pair_found,
        "companion_files_found": companion_ok,
        "best_pair": best_pair,
        "pairs": pairs[:8],
        "gguf_candidates": i2v_ggufs[:40],
        "companions": companion_files,
        "missing": missing,
    }


def _classify_gguf(path: Path) -> dict[str, Any] | None:
    name = path.name.lower()
    if "i2v" not in name:
        return None
    role = None
    if "high" in name:
        role = "high"
    elif "low" in name:
        role = "low"
    if role is None:
        return None
    quant = _extract_quant(path.name)
    return {
        "path": str(path),
        "name": path.name,
        "role": role,
        "quant": quant,
        "size_gb": _safe_size_gb(path),
        "mtime": _safe_mtime(path),
    }


def _extract_quant(name: str) -> str:
    match = GGUF_QUANT_RE.search(name)
    if not match:
        return "unknown"
    return match.group(1).upper()


def _build_pairs(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    highs: dict[str, list[dict[str, Any]]] = {}
    lows: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        quant = str(row.get("quant") or "unknown").upper()
        if row.get("role") == "high":
            highs.setdefault(quant, []).append(row)
        elif row.get("role") == "low":
            lows.setdefault(quant, []).append(row)

    pairs: list[dict[str, Any]] = []
    all_quants = set(highs.keys()) | set(lows.keys())
    for quant in all_quants:
        if quant not in highs or quant not in lows:
            continue
        high = sorted(highs[quant], key=lambda r: (float(r.get("mtime") or 0.0), str(r.get("name"))), reverse=True)[0]
        low = sorted(lows[quant], key=lambda r: (float(r.get("mtime") or 0.0), str(r.get("name"))), reverse=True)[0]
        pairs.append(
            {
                "quant": quant,
                "high": high,
                "low": low,
                "total_size_gb": round(
                    float(high.get("size_gb") or 0.0) + float(low.get("size_gb") or 0.0),
                    2,
                ),
            }
        )
    pairs.sort(key=_pair_sort_key)
    return pairs


def _pair_sort_key(pair: dict[str, Any]) -> tuple[int, int, str]:
    quant = str(pair.get("quant") or "").upper()
    # Prefer common usable mids first (Q4/Q5), then higher, then lower.
    rank_map = {
        "Q4_K_M": 0,
        "Q5_K_M": 1,
        "Q4_K_S": 2,
        "Q5_K_S": 3,
        "Q6_K": 4,
        "Q8_0": 5,
        "Q3_K_M": 6,
        "Q3_K_L": 7,
        "Q3_K_S": 8,
        "Q2_K": 9,
    }
    return (rank_map.get(quant, 99), -int(round(float(pair.get("total_size_gb") or 0.0) * 100)), quant)


def _safe_size_gb(path: Path) -> float:
    try:
        return round(path.stat().st_size / (1024**3), 2)
    except Exception:
        return 0.0


def _safe_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _load_jobs() -> list[dict[str, Any]]:
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


def _save_jobs(rows: list[dict[str, Any]]) -> None:
    STATE_DIR.mkdir(exist_ok=True, parents=True)
    JOBS_FILE.write_text(json.dumps(rows[-500:], indent=2), encoding="utf-8")


def _merge_jobs(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    existing = _load_jobs()
    by_id: dict[str, dict[str, Any]] = {}
    ordered: list[str] = []
    for row in existing:
        pid = str(row.get("prompt_id", "")).strip()
        if not pid:
            continue
        if pid not in by_id:
            ordered.append(pid)
        by_id[pid] = row
    for row in rows:
        pid = str(row.get("prompt_id", "")).strip()
        if not pid:
            continue
        merged = dict(by_id.get(pid, {}))
        merged.update(row)
        by_id[pid] = merged
        if pid not in ordered:
            ordered.append(pid)
    result = [by_id[pid] for pid in ordered if pid in by_id]
    _save_jobs(result)


def _enrich_job_from_history(job: dict[str, Any], config: dict[str, str]) -> dict[str, Any]:
    prompt_id = str(job.get("prompt_id", "")).strip()
    if not prompt_id:
        return job

    base = _configured_comfy_base(config)
    if not base:
        next_job = dict(job)
        next_job["history_error"] = "ComfyUI URL not configured."
        return next_job
    try:
        history_payload = _comfy_request_json(base, "GET", f"/history/{urllib_parse.quote(prompt_id)}", timeout_s=10)
        queue_payload = _comfy_request_json(base, "GET", "/queue", timeout_s=5)
    except Exception as exc:
        next_job = dict(job)
        next_job["history_error"] = str(exc)
        return next_job

    next_job = dict(job)
    history_item = None
    if isinstance(history_payload, dict):
        history_item = history_payload.get(prompt_id)
        if history_item is None and len(history_payload) == 1:
            only_val = next(iter(history_payload.values()))
            if isinstance(only_val, dict):
                history_item = only_val
    if isinstance(history_item, dict):
        outputs = _flatten_comfy_outputs(history_item)
        next_job["status"] = "completed" if outputs else "completed_no_outputs"
        next_job["completed"] = True
        next_job["history_error"] = None
        next_job["history_outputs"] = outputs
        next_job["history_raw"] = {
            "status": history_item.get("status"),
            "meta_keys": sorted(list(history_item.keys())),
        }
        return next_job

    queued_ids = _extract_queue_prompt_ids(queue_payload)
    if prompt_id in queued_ids["running"]:
        next_job["status"] = "running"
    elif prompt_id in queued_ids["queued"]:
        next_job["status"] = "queued"
    else:
        next_job["status"] = next_job.get("status") or "unknown"
    next_job["history_outputs"] = next_job.get("history_outputs", [])
    next_job["history_error"] = None
    return next_job


def _extract_queue_prompt_ids(queue_payload: Any) -> dict[str, set[str]]:
    running: set[str] = set()
    queued: set[str] = set()

    def _walk(rows: Any, bucket: set[str]) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if isinstance(row, dict):
                pid = str(row.get("prompt_id") or row.get("id") or "").strip()
                if pid:
                    bucket.add(pid)
                continue
            if isinstance(row, (list, tuple)):
                for cell in row:
                    if isinstance(cell, str) and cell and len(cell) >= 8:
                        bucket.add(cell)
                        break

    if isinstance(queue_payload, dict):
        _walk(queue_payload.get("queue_running"), running)
        _walk(queue_payload.get("queue_pending"), queued)
        _walk(queue_payload.get("running"), running)
        _walk(queue_payload.get("pending"), queued)
    return {"running": running, "queued": queued}


def _flatten_comfy_outputs(history_item: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = history_item.get("outputs")
    rows: list[dict[str, Any]] = []
    if not isinstance(outputs, dict):
        return rows

    for node_id, node_out in outputs.items():
        if not isinstance(node_out, dict):
            continue
        for key, items in node_out.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                filename = str(item.get("filename") or "").strip()
                if not filename:
                    continue
                media_type = _guess_media_kind(filename=filename, payload_type=str(key))
                query = urllib_parse.urlencode(
                    {
                        "filename": filename,
                        "subfolder": str(item.get("subfolder") or ""),
                        "type": str(item.get("type") or "output"),
                    }
                )
                rows.append(
                    {
                        "node_id": str(node_id),
                        "bucket": str(key),
                        "filename": filename,
                        "subfolder": str(item.get("subfolder") or ""),
                        "type": str(item.get("type") or "output"),
                        "media_kind": media_type,
                        "url": f"./comfy/view?{query}",
                    }
                )
    rows.sort(key=lambda r: (r["media_kind"], r["filename"]))
    return rows


def _guess_media_kind(*, filename: str, payload_type: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".mp4", ".webm", ".mov", ".mkv"}:
        return "video"
    if ext in {".gif"} or "gif" in (payload_type or "").lower():
        return "video"
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    return "file"


def _require_comfy_json(config: dict[str, str], method: str, path: str, payload: Any | None = None) -> dict[str, Any]:
    base = _configured_comfy_base(config)
    if not base:
        raise HTTPException(status_code=400, detail="ComfyUI URL is not configured.")
    try:
        result = _comfy_request_json(base, method, path, payload)
        if not isinstance(result, dict):
            raise HTTPException(status_code=502, detail="ComfyUI returned a non-object JSON payload.")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ComfyUI request failed: {exc}") from exc


def _comfy_request_json(
    base_url: str,
    method: str,
    path: str,
    payload: Any | None = None,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    url = f"{base_url.rstrip('/')}{path}"
    req = urllib_request.Request(url=url, data=body, headers=headers, method=method.upper())
    try:
        with urllib_request.urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise RuntimeError("JSON response was not an object.")
            return data
    except urllib_error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"HTTP {exc.code}: {body_text or exc.reason}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(str(exc.reason or exc)) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from ComfyUI: {exc}") from exc


def _comfy_request_raw(base_url: str, method: str, path: str, timeout_s: float = 20.0) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    req = urllib_request.Request(url=url, headers={}, method=method.upper())
    try:
        with urllib_request.urlopen(req, timeout=float(timeout_s)) as resp:
            return {
                "body": resp.read(),
                "content_type": resp.headers.get("Content-Type", ""),
                "content_disposition": resp.headers.get("Content-Disposition", ""),
            }
    except urllib_error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"HTTP {exc.code}: {body_text or exc.reason}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(str(exc.reason or exc)) from exc


def _staging_item(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "web_path": f"/{to_web_path(path)}",
        "size_bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
    }


def _safe_stem(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    return clean.strip("._-")[:80]


def _recent_webbduck_images(limit: int = 24) -> list[dict[str, Any]]:
    runs = []
    try:
        runs = [p for p in BASE.iterdir() if p.is_dir() and p.name != STAGING_DIR.name]
    except Exception:
        return []
    runs.sort(key=lambda p: p.name, reverse=True)

    out: list[dict[str, Any]] = []
    for run in runs:
        for img in sorted(run.glob("*.png"), key=lambda p: p.name):
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
            if len(out) >= int(limit):
                return out
    return out
