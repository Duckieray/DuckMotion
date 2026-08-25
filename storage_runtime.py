"""Architecture-neutral persistence and filesystem services for DuckMotion."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from server.storage import BASE, resolve_web_path, to_web_path


STATE_DIR = Path.home() / ".webbduck" / "plugin_state"
CONFIG_FILE = STATE_DIR / "duckmotion_config.json"
JOBS_FILE = STATE_DIR / "duckmotion_jobs.json"
DEFAULT_OUTPUT_DIR = BASE / "duckmotion_videos"
STAGING_DIR = BASE / "duckmotion_staging"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv"}


class VideoStorageRuntime:
    """Own DuckMotion state without depending on a video architecture backend."""

    def __init__(
        self,
        *,
        config_file: Path = CONFIG_FILE,
        jobs_file: Path = JOBS_FILE,
        staging_dir: Path = STAGING_DIR,
        default_output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        self.config_file = Path(config_file)
        self.jobs_file = Path(jobs_file)
        self.staging_dir = Path(staging_dir)
        self.default_output_dir = Path(default_output_dir)
        self._lock = threading.RLock()

    @staticmethod
    def now() -> float:
        return time.time()

    def load_config(self) -> dict[str, Any]:
        raw: dict[str, Any] = {}
        if self.config_file.exists():
            try:
                value = json.loads(self.config_file.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    raw = value
            except Exception:
                raw = {}
        return {
            "model_id_or_path": str(
                raw.get("model_id_or_path")
                if "model_id_or_path" in raw
                else os.getenv("DUCKMOTION_MODEL_ID_OR_PATH", "")
            ).strip(),
            "models_dir": str(
                raw.get("models_dir")
                if "models_dir" in raw
                else os.getenv("DUCKMOTION_MODELS_DIR", "")
            ).strip(),
            "output_dir": str(
                raw.get("output_dir")
                if "output_dir" in raw
                else os.getenv("DUCKMOTION_OUTPUT_DIR", "")
            ).strip(),
        }

    def save_config(self, config: dict[str, Any]) -> None:
        self.config_file.parent.mkdir(exist_ok=True, parents=True)
        payload = {
            "model_id_or_path": str(config.get("model_id_or_path") or "").strip(),
            "models_dir": str(config.get("models_dir") or "").strip(),
            "output_dir": str(config.get("output_dir") or "").strip(),
        }
        self.config_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def resolve_output_dir(self, config: dict[str, Any]) -> Path:
        raw = str(config.get("output_dir") or "").strip()
        path = Path(raw).expanduser() if raw else self.default_output_dir
        path.mkdir(exist_ok=True, parents=True)
        return path

    def output_writable(self, config: dict[str, Any]) -> tuple[bool, str | None]:
        path = self.resolve_output_dir(config)
        probe = path / f".duckmotion-write-{os.getpid()}-{threading.get_ident()}"
        try:
            probe.write_bytes(b"")
            probe.unlink(missing_ok=True)
            return True, None
        except Exception as exc:
            try:
                probe.unlink(missing_ok=True)
            except Exception:
                pass
            return False, str(exc)

    @staticmethod
    def _validate_image_path(path: Path) -> Path:
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported input image type: {path.suffix}")
        return path

    def resolve_input_image(self, raw_path: str) -> Path:
        raw = str(raw_path or "").strip()
        if not raw:
            raise ValueError("Input image path is empty")
        direct = Path(raw).expanduser()
        if direct.exists() and direct.is_file():
            return self._validate_image_path(direct.resolve())
        try:
            resolved = Path(resolve_web_path(raw)).resolve()
        except Exception as exc:
            raise FileNotFoundError(f"Input image not found: {raw}") from exc
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"Input image not found: {raw}")
        return self._validate_image_path(resolved)

    @staticmethod
    def safe_web_path(path: Path) -> str | None:
        try:
            raw = to_web_path(Path(path))
            return "/" + str(raw).lstrip("/")
        except Exception:
            return None

    def _load_jobs_locked(self) -> list[dict[str, Any]]:
        if not self.jobs_file.exists():
            return []
        try:
            raw = json.loads(self.jobs_file.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [dict(row) for row in raw if isinstance(row, dict)] if isinstance(raw, list) else []

    def _save_jobs_locked(self, rows: list[dict[str, Any]]) -> None:
        self.jobs_file.parent.mkdir(exist_ok=True, parents=True)
        tmp = self.jobs_file.with_suffix(self.jobs_file.suffix + ".tmp")
        tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        tmp.replace(self.jobs_file)

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._load_jobs_locked()
        rows.sort(key=lambda row: float(row.get("created_at") or 0.0), reverse=True)
        return rows[: max(1, int(limit))]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        target = str(job_id or "").strip()
        if not target:
            return None
        with self._lock:
            for row in self._load_jobs_locked():
                if str(row.get("job_id") or "") == target:
                    return row
        return None

    def upsert_job(self, job: dict[str, Any]) -> dict[str, Any]:
        job_id = str(job.get("job_id") or "").strip()
        if not job_id:
            raise ValueError("Job must have a job_id")
        value = dict(job)
        with self._lock:
            rows = self._load_jobs_locked()
            for index, row in enumerate(rows):
                if str(row.get("job_id") or "") == job_id:
                    rows[index] = value
                    break
            else:
                rows.append(value)
            self._save_jobs_locked(rows)
        return value

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        target = str(job_id or "").strip()
        with self._lock:
            rows = self._load_jobs_locked()
            for row in rows:
                if str(row.get("job_id") or "") != target:
                    continue
                if row.get("status") in {"completed", "failed", "canceled"}:
                    return {"ok": True, "job": dict(row)}
                row["cancel_requested"] = True
                row["updated_at"] = self.now()
                if row.get("status") == "queued":
                    row["status"] = "canceled"
                    row["finished_at"] = self.now()
                    row["progress"] = {"stage": "canceled", "percent": 0}
                self._save_jobs_locked(rows)
                return {"ok": True, "job": dict(row)}
        return {"ok": False, "error": "Job not found."}

    def clear_jobs(self) -> None:
        with self._lock:
            self._save_jobs_locked([])

    def queue_snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = self._load_jobs_locked()
        return {
            "queued": sum(1 for row in rows if row.get("status") == "queued"),
            "running": sum(1 for row in rows if row.get("status") == "running"),
            "cancel_requested": sum(1 for row in rows if row.get("cancel_requested")),
        }

    def staging_item(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        return {
            "name": path.name,
            "path": str(path),
            "web_path": self.safe_web_path(path),
            "size": path.stat().st_size if path.exists() else 0,
            "modified_at": path.stat().st_mtime if path.exists() else 0.0,
        }

    def list_staging(self, limit: int = 100) -> list[dict[str, Any]]:
        self.staging_dir.mkdir(exist_ok=True, parents=True)
        rows = [
            path
            for path in self.staging_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]
        rows.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return [self.staging_item(path) for path in rows[: max(1, int(limit))]]

    def stage_bytes(self, filename: str, data: bytes) -> dict[str, Any]:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError("Unsupported image type.")
        if not data:
            raise ValueError("Empty upload.")
        self.staging_dir.mkdir(exist_ok=True, parents=True)
        stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in Path(filename).stem).strip("_") or "input"
        path = self.staging_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() & 0xFFFFFFFF:08x}_{stem}{suffix}"
        path.write_bytes(data)
        return self.staging_item(path)

    def stage_from_webbduck(self, raw_path: str) -> dict[str, Any]:
        source = self.resolve_input_image(raw_path)
        self.staging_dir.mkdir(exist_ok=True, parents=True)
        path = self.staging_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() & 0xFFFFFFFF:08x}_{source.name}"
        path.write_bytes(source.read_bytes())
        return self.staging_item(path)

    def delete_staging(self, name: str) -> str:
        safe_name = Path(str(name or "")).name
        if not safe_name:
            raise ValueError("Invalid file name.")
        path = self.staging_dir / safe_name
        if not path.exists() or not path.is_file():
            raise FileNotFoundError("Staged file not found.")
        path.unlink()
        return safe_name

    def gallery_item_from_run(self, run_dir: Path) -> dict[str, Any] | None:
        run_dir = Path(run_dir)
        if not run_dir.exists() or not run_dir.is_dir():
            return None
        meta: dict[str, Any] = {}
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            try:
                value = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    meta = value
            except Exception:
                meta = {}
        videos = sorted(
            [path for path in run_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES],
            key=lambda path: path.name.lower(),
        )
        if not videos:
            return None
        video = videos[0]
        poster = next(
            (run_dir / name for name in ("poster.jpg", "poster.png", "preview.jpg", "preview.png") if (run_dir / name).exists()),
            None,
        )
        run_id = str(meta.get("run_id") or run_dir.name)
        return {
            "run_id": run_id,
            "job_id": meta.get("job_id"),
            "created_at": meta.get("created_at") or run_dir.stat().st_mtime,
            "video": f"/gallery/file/{run_id}/{video.name}",
            "poster": f"/gallery/file/{run_id}/{poster.name}" if poster is not None else None,
            "video_path": str(video),
            "poster_path": str(poster) if poster is not None else None,
            "meta": meta,
        }

    def scan_gallery(self, config: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
        root = self.resolve_output_dir(config)
        dirs = [path for path in root.iterdir() if path.is_dir()]
        dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        items: list[dict[str, Any]] = []
        for run_dir in dirs:
            item = self.gallery_item_from_run(run_dir)
            if item is not None:
                items.append(item)
            if len(items) >= max(1, int(limit)):
                break
        return items


storage_runtime = VideoStorageRuntime()
