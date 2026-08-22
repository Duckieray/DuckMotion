"""Wan 2.2 backend behind DuckMotion's generic VideoBackend contract."""

from __future__ import annotations

from typing import Any

from model_runtime import VideoBackend, VideoModelDescriptor, backend_resolver


class WanDiffusersBackend(VideoBackend):
    backend_id = "wan_diffusers"

    def __init__(self, implementation: Any) -> None:
        self._impl = implementation

    def can_handle(self, descriptor: VideoModelDescriptor) -> bool:
        # The migrated Wan runtime currently owns the proven image-to-video
        # implementation only. A future Wan T2V adapter can register separately
        # without making the generic router branch on family names.
        return (
            descriptor.backend == self.backend_id
            and descriptor.architecture == "wan22"
            and descriptor.capabilities.image_to_video
        )

    def readiness(self, descriptor: VideoModelDescriptor) -> dict[str, Any]:
        if not self.can_handle(descriptor):
            return {
                "ready": False,
                "reason": f"The installed runtime does not implement this workflow for '{descriptor.name}'.",
            }
        try:
            status = self._impl._probe_diffusers_support()
        except Exception as exc:
            return {"ready": False, "reason": f"Video runtime probe failed: {exc}"}
        if not isinstance(status, dict) or not status.get("ready"):
            reason = status.get("error") if isinstance(status, dict) else None
            return {
                "ready": False,
                "reason": str(reason or "Required video Diffusers pipeline is unavailable."),
            }
        return {"ready": True, "reason": None}

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
