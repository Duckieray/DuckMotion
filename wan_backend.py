"""Wan 2.2 backend behind DuckMotion's generic VideoBackend contract."""

from __future__ import annotations

from typing import Any

from model_runtime import VideoBackend, VideoModelDescriptor, backend_resolver


class WanDiffusersBackend(VideoBackend):
    backend_id = "wan_diffusers"

    def __init__(self, implementation: Any) -> None:
        self._impl = implementation

    def can_handle(self, descriptor: VideoModelDescriptor) -> bool:
        return descriptor.backend == self.backend_id and descriptor.architecture == "wan22"

    def generate(
        self,
        descriptor: VideoModelDescriptor,
        request: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        job = kwargs.get("job")
        config = kwargs.get("config")
        runtime_profile = kwargs.get("runtime_profile")
        lease_token = kwargs.get("lease_token")
        if not isinstance(job, dict) or not isinstance(config, dict) or not isinstance(runtime_profile, dict):
            raise RuntimeError("Wan backend requires job, config, and runtime profile context")

        self._impl._prepare_runtime_for_wan(runtime_profile)
        frames = self._impl._generate_frames_with_diffusers(
            job,
            config,
            runtime_profile,
            persist_progress=False,
            lease_token=lease_token,
        )
        output_info = self._impl._write_video_outputs(job, frames, config)
        return {"output_info": output_info}

    def unload(self) -> None:
        self._impl._unload_pipeline()


_backend: WanDiffusersBackend | None = None


def ensure_registered(implementation: Any) -> WanDiffusersBackend:
    global _backend
    if _backend is None or _backend._impl is not implementation:
        _backend = WanDiffusersBackend(implementation)
    backend_resolver.register(_backend)
    return _backend
