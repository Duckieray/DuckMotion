"""Architecture-neutral runtime services for DuckMotion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import host_runtime
from storage_runtime import VideoStorageRuntime, storage_runtime


class VideoRuntimeServices:
    """Compose generic storage with WebbDuck host runtime primitives."""

    def __init__(self, storage: VideoStorageRuntime | None = None) -> None:
        self.storage = storage or storage_runtime

    def now(self) -> float:
        return self.storage.now()

    def load_config(self) -> dict[str, Any]:
        return self.storage.load_config()

    def save_config(self, config: dict[str, Any]) -> None:
        self.storage.save_config(config)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.storage.get_job(job_id)

    def upsert_job(self, job: dict[str, Any]) -> dict[str, Any]:
        return self.storage.upsert_job(job)

    def resolve_input_image(self, path: str) -> Path:
        return self.storage.resolve_input_image(path)

    def resolve_output_dir(self, config: dict[str, Any]) -> Path:
        return self.storage.resolve_output_dir(config)

    def safe_web_path(self, path: Path) -> str | None:
        return self.storage.safe_web_path(path)

    def gallery_item_from_run(self, run_dir: Path) -> dict[str, Any] | None:
        return self.storage.gallery_item_from_run(run_dir)

    def runtime_profile(self) -> dict[str, Any]:
        return host_runtime.runtime_profile_or_raise()

    def runtime_profile_safe(self) -> dict[str, Any]:
        return host_runtime.runtime_profile_safe()

    def gpu_lease(self) -> dict[str, Any]:
        return host_runtime.gpu_lease()

    def queue_snapshot(self) -> dict[str, Any]:
        return self.storage.queue_snapshot()

    def output_writable(self, config: dict[str, Any]) -> tuple[bool, str | None]:
        return self.storage.output_writable(config)

    def acquire_gpu_lease(self, **kwargs: Any) -> dict[str, Any]:
        return host_runtime.acquire_gpu_lease(**kwargs)

    def release_gpu_lease(self, **kwargs: Any) -> Any:
        return host_runtime.release_gpu_lease(**kwargs)

    def lease_heartbeat(self, **kwargs: Any) -> bool:
        return host_runtime.lease_heartbeat(**kwargs)
