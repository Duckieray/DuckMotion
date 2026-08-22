"""Adapter exposing shared DuckMotion services to architecture-neutral code."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class VideoRuntimeServices:
    """Temporary façade over generic utilities still housed in backend.py.

    Public routing, health, config, and job coordination depend on this neutral
    façade rather than on Wan-shaped module globals. As the underlying helpers
    are physically extracted, this adapter can disappear without changing the
    model-facing contracts.
    """

    def __init__(self, implementation: Any) -> None:
        self._impl = implementation

    def now(self) -> float:
        return float(self._impl._now())

    def load_config(self) -> dict[str, Any]:
        return dict(self._impl._load_config())

    def save_config(self, config: dict[str, Any]) -> None:
        self._impl._save_config(dict(config))

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

    def runtime_profile_safe(self) -> dict[str, Any]:
        return dict(self._impl._get_runtime_profile_safe())

    def gpu_lease(self) -> dict[str, Any]:
        value = self._impl.get_gpu_lease()
        return dict(value) if isinstance(value, dict) else {"held": False}

    def queue_snapshot(self) -> dict[str, Any]:
        value = self._impl._queue_snapshot()
        return dict(value) if isinstance(value, dict) else {}

    def output_writable(self, config: dict[str, Any]) -> tuple[bool, str | None]:
        return self._impl._probe_writable_dir(self.resolve_output_dir(config))

    def acquire_gpu_lease(self, **kwargs: Any) -> dict[str, Any]:
        return self._impl.acquire_gpu_lease_blocking(**kwargs)

    def release_gpu_lease(self, **kwargs: Any) -> Any:
        return self._impl.release_gpu_lease(**kwargs)

    def unload_wan_pipeline(self) -> None:
        self._impl._unload_pipeline()
