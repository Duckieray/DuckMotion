from __future__ import annotations

from pathlib import Path

from job_runtime import VideoJobCoordinator
from model_runtime import describe_video_model


class _Services:
    def resolve_input_image(self, path):
        return Path(path)


def test_ltx_params_use_model_constraints_and_defaults():
    coordinator = VideoJobCoordinator(_Services())
    descriptor = describe_video_model("Lightricks/LTX-2.5-Diffusers")
    params = coordinator._normalize_params(
        descriptor,
        {
            "prompt": "test",
            "width": 777,
            "height": 530,
            "num_frames": 120,
        },
    )
    assert params["width"] % 32 == 0
    assert params["height"] % 32 == 0
    assert params["num_frames"] % 8 == 1
    assert params["fps"] == 24
    assert params["guidance_scale"] == 1.0
    assert "negative_prompt" not in params


def test_wan_params_use_model_defaults_without_wan_specific_coordinator_code():
    coordinator = VideoJobCoordinator(_Services())
    descriptor = describe_video_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    params = coordinator._normalize_params(
        descriptor,
        {"prompt": "move", "negative_prompt": "blur"},
    )
    assert params["width"] == 832
    assert params["height"] == 480
    assert params["num_frames"] == 81
    assert params["fps"] == 16
    assert params["num_inference_steps"] == 30
    assert params["guidance_scale"] == 5.0
    assert params["negative_prompt"] == "blur"


def test_parameter_normalization_uses_descriptor_constraints_not_family_name():
    import inspect

    source = inspect.getsource(VideoJobCoordinator._normalize_params)
    assert "wan22" not in source
    assert "ltx25" not in source
    assert "architecture" not in source
