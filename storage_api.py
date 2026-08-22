"""Generic DuckMotion job, staging, and gallery API surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from storage_runtime import VideoStorageRuntime


class CancelPayload(BaseModel):
    job_id: str


def build_storage_router(
    storage: VideoStorageRuntime,
    load_config: Callable[[], dict[str, Any]],
) -> APIRouter:
    router = APIRouter()

    @router.get("/engine/jobs")
    def engine_jobs(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        return {"jobs": storage.list_jobs(int(limit))}

    @router.get("/engine/jobs/{job_id}")
    def engine_job(job_id: str) -> dict[str, Any]:
        row = storage.get_job(job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return {"job": row}

    @router.post("/engine/cancel")
    def engine_cancel(payload: CancelPayload) -> dict[str, Any]:
        result = storage.cancel_job(payload.job_id)
        if not result.get("ok"):
            raise HTTPException(status_code=404, detail=str(result.get("error") or "Job not found."))
        return result

    @router.post("/jobs/clear")
    def clear_jobs() -> dict[str, Any]:
        storage.clear_jobs()
        return {"ok": True}

    @router.post("/staging/upload")
    async def staging_upload(image: UploadFile = File(...)) -> dict[str, Any]:
        filename = str(image.filename or "").strip()
        if not filename:
            raise HTTPException(status_code=400, detail="Missing filename.")
        try:
            item = storage.stage_bytes(filename, await image.read())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "item": item}

    @router.post("/staging/from-webbduck")
    def staging_from_webbduck(path: str = Form(...)) -> dict[str, Any]:
        try:
            item = storage.stage_from_webbduck(path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "item": item}

    @router.get("/staging")
    def list_staging(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        return {"items": storage.list_staging(int(limit))}

    @router.delete("/staging/{name}")
    def delete_staging(name: str) -> dict[str, Any]:
        try:
            deleted = storage.delete_staging(name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "deleted": deleted}

    @router.get("/gallery")
    def gallery(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        return {"items": storage.scan_gallery(load_config(), int(limit))}

    @router.get("/gallery/file/{run_id}/{filename}")
    def gallery_file(run_id: str, filename: str) -> FileResponse:
        config = load_config()
        root = storage.resolve_output_dir(config).resolve()
        safe_run = Path(str(run_id or "")).name
        safe_name = Path(str(filename or "")).name
        if not safe_run or not safe_name or safe_run != str(run_id) or safe_name != str(filename):
            raise HTTPException(status_code=400, detail="Invalid gallery path.")
        target = (root / safe_run / safe_name).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid gallery path.") from exc
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found.")
        return FileResponse(target)

    return router
