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


def _model_index(root: Path, class_name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": class_name}),
        encoding="utf-8",
    )


def test_detects_wan_i2v_and_requires_source_image(tmp_path):
    _model_index(tmp_path, "WanImageToVideoPipeline")
    arch, caps, detection = detect_video_architecture(str(tmp_path))
    assert arch == "wan22"
    assert caps.image_to_video is True
    assert caps.text_to_video is False
    assert caps.source_image_required is True
    assert detection["method"] == "local_config"


def test_detects_ltx25_with_t2v_i2v_and_audio(tmp_path):
    _model_index(tmp_path, "LTX2ImageToVideoPipeline")
    descriptor = describe_video_model(str(tmp_path), name="LTX-2.5")
    assert descriptor.architecture == "ltx25"
    assert descriptor.backend == "ltx25_isolated"
    assert descriptor.capabilities.text_to_video is True
    assert descriptor.capabilities.image_to_video is True
    assert descriptor.capabilities.audio_output is True
    assert descriptor.capabilities.source_image_required is False
    assert descriptor.constraints["dimension_multiple"] == 32
    assert descriptor.constraints["frame_count_remainder"] == 1
    assert descriptor.supported is False


def test_existing_wan_backend_remains_runnable(tmp_path):
    _model_index(tmp_path, "WanImageToVideoPipeline")
    descriptor = describe_video_model(str(tmp_path), name="Wan2.2")
    assert descriptor.supported is True


def test_unknown_video_model_is_not_assumed_to_be_wan(tmp_path):
    _model_index(tmp_path, "FutureVideoPipeline")
    descriptor = describe_video_model(str(tmp_path))
    assert descriptor.architecture == "unknown"
    assert descriptor.supported is False


def test_public_descriptor_hides_architecture_and_backend():
    descriptor = describe_video_model(
        "Lightricks/LTX-2.5-Diffusers",
        name="LTX-2.5",
        defaults={"fps": 24},
    )
    payload = descriptor.to_public_dict()
    assert payload["capabilities"]["text_to_video"] is True
    assert payload["defaults"]["fps"] == 24
    assert payload["supported"] is False
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
