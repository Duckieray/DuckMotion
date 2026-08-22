"""Adapter exposing shared DuckMotion services to architecture-neutral code."""

from __future__ import annotations

import json
import os
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
        """Load generic saved fields over temporary backend-internal defaults."""
        config = dict(self._impl._default_config())
        raw: dict[str, Any] = {}
        path = Path(self._impl.CONFIG_FILE)
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    raw = value
            except Exception:
                raw = {}

        # Selection is no longer implicitly Wan. With no saved selection, only
        # an explicit environment override selects a model.
        config["model_id_or_path"] = str(
            raw.get("model_id_or_path")
            if "model_id_or_path" in raw
            else os.getenv("DUCKMOTION_MODEL_ID_OR_PATH", "")
        ).strip()
        if "models_dir" in raw:
            config["models_dir"] = str(raw.get("models_dir") or "").strip()
        if "output_dir" in raw:
            config["output_dir"] = str(raw.get("output_dir") or "").strip()
        return config

    def save_config(self, config: dict[str, Any]) -> None:
        """Persist only the architecture-neutral user configuration."""
        path = Path(self._impl.CONFIG_FILE)
        path.parent.mkdir(exist_ok=True, parents=True)
        payload = {
            "model_id_or_path": str(config.get("model_id_or_path") or "").strip(),
            "models_dir": str(config.get("models_dir") or "").strip(),
            "output_dir": str(config.get("output_dir") or "").strip(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
