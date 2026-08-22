from __future__ import annotations

from model_runtime import describe_video_model
from wan_backend import WanDiffusersBackend


class _Implementation:
    def __init__(self):
        self.prepared = None
        self.generated = None
        self.unloaded = False

    def _prepare_runtime_for_wan(self, runtime_profile):
        self.prepared = runtime_profile

    def _generate_frames_with_diffusers(self, job, config, runtime_profile, **kwargs):
        self.generated = {
            "job": job,
            "config": config,
            "runtime_profile": runtime_profile,
            "kwargs": kwargs,
        }
        return ["frame"]

    def _write_video_outputs(self, job, frames, config):
        return {"run_id": "wan-run", "frames": len(frames)}

    def _unload_pipeline(self):
        self.unloaded = True


def test_wan_backend_owns_wan_generation_primitives():
    impl = _Implementation()
    backend = WanDiffusersBackend(impl)
    descriptor = describe_video_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    result = backend.generate(
        descriptor,
        {"prompt": "move"},
        job={"job_id": "dm_1"},
        config={"output_dir": "/tmp"},
        runtime_profile={"device": "cuda"},
        lease_token="lease",
    )

    assert backend.can_handle(descriptor) is True
    assert impl.prepared == {"device": "cuda"}
    assert impl.generated["kwargs"]["persist_progress"] is False
    assert impl.generated["kwargs"]["lease_token"] == "lease"
    assert result["output_info"]["run_id"] == "wan-run"


def test_wan_backend_unload_is_backend_owned():
    impl = _Implementation()
    backend = WanDiffusersBackend(impl)
    backend.unload()
    assert impl.unloaded is True
