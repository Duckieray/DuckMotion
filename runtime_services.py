"""Adapter exposing shared DuckMotion services to architecture-neutral job code."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class VideoRuntimeServices:
    """Temporary façade over shared utilities still housed in backend.py.

    The coordinator and video backends depend on this generic façade rather than
    on Wan-specific module globals. As generic persistence/gallery/GPU modules
    are extracted, this adapter can disappear without changing backend APIs.
    """

    def __init__(self, implementation: Any) -> None:
        self._impl = implementation

    def now(self) -> float:
        return float(self._impl._now())

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._impl._get_job(job_id)

    def upsert_job(self, job: dict[str, Any]) -> dict[str, Any]:
        return self._impl._upsert_job(job)

    def resolve_input_image(self, path: str) -> Path:
        return self._impl._resolve_input_image(path)

    def resolve_output_dir(self, config: dict[str, Any]) -> Path:
        return self._impl._resolve_output_dir(config)

    def safe_web_path(self, path: Path) -> str | None:
        return self._impl._safe_web_path_for_any(path)

    def gallery_item_from_run(self, run_dir: Path) -> dict[str, Any] | None:
        return self._impl._gallery_item_from_run(run_dir)

    def runtime_profile(self) -> dict[str, Any]:
        return self._impl._get_runtime_profile_or_raise()

    def acquire_gpu_lease(self, **kwargs: Any) -> dict[str, Any]:
        return self._impl.acquire_gpu_lease_blocking(**kwargs)

    def release_gpu_lease(self, **kwargs: Any) -> Any:
        return self._impl.release_gpu_lease(**kwargs)

    def unload_wan_pipeline(self) -> None:
        self._impl._unload_pipeline()
