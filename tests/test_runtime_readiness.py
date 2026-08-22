from __future__ import annotations

from ltx_backend import LTX25IsolatedBackend
from model_runtime import VideoBackendResolver, describe_video_model
from wan_backend import WanDiffusersBackend


class _Result:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stdout = ""
        self.stderr = stderr


def test_wan_t2v_is_now_runnable_by_isolated_backend():
    descriptor = describe_video_model("Wan-AI/Wan2.2-T2V-A14B-Diffusers")

    assert descriptor.capabilities.text_to_video is True
    assert descriptor.capabilities.image_to_video is False
    assert descriptor.backend == "wan_diffusers"
    assert descriptor.supported is True


def test_wan_readiness_probes_isolated_python_and_reports_failure(monkeypatch):
    descriptor = describe_video_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    backend = WanDiffusersBackend()

    monkeypatch.setattr(
        "wan_backend.subprocess.run",
        lambda *args, **kwargs: _Result(returncode=1, stderr="ImportError: pipeline missing"),
    )
    readiness = backend.readiness(descriptor)

    assert readiness["ready"] is False
    assert "pipeline missing" in readiness["reason"]


def test_resolver_readiness_delegates_to_selected_wan_backend(monkeypatch):
    descriptor = describe_video_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    backend = WanDiffusersBackend()
    monkeypatch.setattr(
        "wan_backend.subprocess.run",
        lambda *args, **kwargs: _Result(returncode=0),
    )
    resolver = VideoBackendResolver()
    resolver.register(backend)

    assert resolver.readiness(descriptor) == {"ready": True, "reason": None}


def test_wan_readiness_is_cached(monkeypatch):
    descriptor = describe_video_model("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    backend = WanDiffusersBackend()
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _Result(returncode=0)

    monkeypatch.setattr("wan_backend.subprocess.run", fake_run)
    first = backend.readiness(descriptor)
    second = backend.readiness(descriptor)

    assert first == {"ready": True, "reason": None}
    assert second == first
    assert len(calls) == 1


def test_ltx_readiness_probes_two_stage_runtime_and_caches_result(monkeypatch):
    descriptor = describe_video_model("Lightricks/LTX-2.5-Diffusers")
    backend = LTX25IsolatedBackend()
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _Result(returncode=0)

    monkeypatch.setattr("ltx_backend.subprocess.run", fake_run)

    first = backend.readiness(descriptor)
    second = backend.readiness(descriptor)

    assert first == {"ready": True, "reason": None}
    assert second == first
    assert len(calls) == 1
    assert calls[0][0][1] == "-c"
    assert "LTX2LatentUpsamplePipeline" in calls[0][0][2]
    assert "STAGE_2_DISTILLED_SIGMA_VALUES" in calls[0][0][2]
