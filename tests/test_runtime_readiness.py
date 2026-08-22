from __future__ import annotations

from ltx_backend import LTX25IsolatedBackend
from model_runtime import VideoBackendResolver, describe_video_model
from wan_backend import WanDiffusersBackend


class _WanImplementation:
    def __init__(self, ready=True):
        self.ready = ready

    def _probe_diffusers_support(self):
        return {"ready": self.ready, "error": None if self.ready else "pipeline missing"}

    def _unload_pipeline(self):
        pass


def test_wan_t2v_is_detected_but_not_advertised_as_runnable():
    descriptor = describe_video_model("Wan-AI/Wan2.2-T2V-A14B-Diffusers")

    assert descriptor.capabilities.text_to_video is True
    assert descriptor.capabilities.image_to_video is False
    assert descriptor.backend == "unsupported"
    assert descriptor.supported is False


def test_wan_i2v_readiness_uses_installed_pipeline_probe():
    descriptor = describe_video_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    backend = WanDiffusersBackend(_WanImplementation(ready=False))

    readiness = backend.readiness(descriptor)

    assert readiness["ready"] is False
    assert "pipeline missing" in readiness["reason"]


def test_resolver_readiness_delegates_to_selected_backend():
    descriptor = describe_video_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    resolver = VideoBackendResolver()
    resolver.register(WanDiffusersBackend(_WanImplementation(ready=True)))

    assert resolver.readiness(descriptor) == {"ready": True, "reason": None}


def test_ltx_readiness_probes_isolated_python_and_caches_result(monkeypatch):
    descriptor = describe_video_model("Lightricks/LTX-2.5-Diffusers")
    backend = LTX25IsolatedBackend()
    calls = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return Result()

    monkeypatch.setattr("ltx_backend.subprocess.run", fake_run)

    first = backend.readiness(descriptor)
    second = backend.readiness(descriptor)

    assert first == {"ready": True, "reason": None}
    assert second == first
    assert len(calls) == 1
    assert calls[0][0][1] == "-c"
