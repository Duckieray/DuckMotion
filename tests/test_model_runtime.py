import json
from pathlib import Path

import pytest

from model_runtime import (
    VideoBackend,
    VideoBackendResolver,
    VideoModelDescriptor,
    describe_video_model,
    detect_video_architecture,
)


class _Backend(VideoBackend):
    backend_id = "test_video_backend"

    def can_handle(self, descriptor: VideoModelDescriptor) -> bool:
        return descriptor.backend == self.backend_id

    def generate(self, descriptor, request, **kwargs):
        return {"model": descriptor.name, "request": request}


def _model_index(root: Path, class_name: str, **extra) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": class_name, **extra}),
        encoding="utf-8",
    )


def test_detects_wan_i2v_and_requires_source_image(tmp_path):
    _model_index(tmp_path, "WanImageToVideoPipeline")
    arch, caps, detection = detect_video_architecture(str(tmp_path))
    assert arch == "wan22"
    assert caps.image_to_video is True
    assert caps.text_to_video is False
    assert caps.source_image_required is True
    assert detection["variant"] == "i2v"


def test_detects_wan_t2v_as_runnable(tmp_path):
    _model_index(tmp_path, "WanPipeline")
    descriptor = describe_video_model(str(tmp_path), name="Wan2.2-T2V-A14B-Diffusers")
    assert descriptor.backend == "wan_diffusers"
    assert descriptor.capabilities.text_to_video is True
    assert descriptor.capabilities.image_to_video is False
    assert descriptor.capabilities.source_image_required is False
    assert descriptor.supported is True


def test_wan_ti2v_tracks_upstream_i2v_but_only_advertises_current_diffusers_workflow(tmp_path):
    _model_index(tmp_path, "WanPipeline")
    descriptor = describe_video_model(
        str(tmp_path),
        name="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    )
    assert descriptor.capabilities.text_to_video is True
    assert descriptor.capabilities.image_to_video is False
    assert descriptor.capabilities.source_image_required is False
    assert descriptor.detection["variant"] == "ti2v"
    assert descriptor.detection["upstream_image_to_video"] is True
    assert descriptor.detection["runtime_image_to_video"] is False
    assert descriptor.defaults == {
        "width": 1280,
        "height": 704,
        "num_frames": 121,
        "fps": 24,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
    }
    assert descriptor.supported is True


def test_detects_ltx25_with_live_t2v_i2v_audio_and_two_stage_constraints(tmp_path):
    _model_index(tmp_path, "LTX2ImageToVideoPipeline")
    descriptor = describe_video_model(str(tmp_path), name="LTX-2.5")
    assert descriptor.architecture == "ltx25"
    assert descriptor.backend == "ltx25_isolated"
    assert descriptor.capabilities.text_to_video is True
    assert descriptor.capabilities.image_to_video is True
    assert descriptor.capabilities.video_to_video is False
    assert descriptor.capabilities.audio_output is True
    assert descriptor.capabilities.negative_prompt is False
    assert descriptor.capabilities.source_image_required is False
    assert descriptor.constraints["dimension_multiple"] == 64
    assert descriptor.constraints["generation_stages"] == 2
    assert descriptor.constraints["sampling_schedule_locked"] is True
    assert descriptor.constraints["frame_count_remainder"] == 1
    assert descriptor.defaults["width"] == 1536
    assert descriptor.defaults["height"] == 1024
    assert descriptor.defaults["num_frames"] == 121
    assert descriptor.defaults["fps"] == 24
    assert descriptor.supported is True


def test_wan_turbo_defaults_are_checkpoint_specific():
    descriptor = describe_video_model("Wan-AI/Wan2.2-Turbo-TI2V-5B-Diffusers")
    assert descriptor.defaults["width"] == 1280
    assert descriptor.defaults["height"] == 704
    assert descriptor.defaults["num_inference_steps"] == 4
    assert descriptor.defaults["guidance_scale"] == 1.0
    assert descriptor.defaults["fps"] == 24
    assert descriptor.defaults["num_frames"] == 121


def test_unknown_video_model_is_not_assumed_to_be_wan(tmp_path):
    _model_index(tmp_path, "FutureVideoPipeline")
    descriptor = describe_video_model(str(tmp_path))
    assert descriptor.architecture == "unknown"
    assert descriptor.supported is False


def test_public_descriptor_hides_architecture_and_backend():
    descriptor = describe_video_model(
        "Lightricks/LTX-2.5-Diffusers",
        name="LTX-2.5",
    )
    payload = descriptor.to_public_dict()
    assert payload["capabilities"]["text_to_video"] is True
    assert payload["defaults"]["fps"] == 24
    assert payload["constraints"]["generation_stages"] == 2
    assert payload["supported"] is True
    assert "architecture" not in payload
    assert "backend" not in payload


def test_video_backend_resolver_routes_without_ui_engine_choice():
    resolver = VideoBackendResolver()
    resolver.register(_Backend())
    descriptor = VideoModelDescriptor(
        name="future",
        source="future/model",
        architecture="future",
        backend="test_video_backend",
    )
    assert resolver.resolve(descriptor).backend_id == "test_video_backend"


def test_video_backend_resolver_fails_actionably():
    resolver = VideoBackendResolver()
    descriptor = VideoModelDescriptor(name="future", source="future/model")
    with pytest.raises(LookupError, match="future"):
        resolver.resolve(descriptor)
