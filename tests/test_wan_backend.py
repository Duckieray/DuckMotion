from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_runtime import describe_video_model
from wan_backend import WanDiffusersBackend
from wan_worker import _snap_dimension, _snap_frames


def _fake_process_factory(captured, tmp_path):
    class FakeProcess:
        def __init__(self, cmd, **_kwargs):
            request_path = Path(cmd[cmd.index("--request") + 1])
            result_path = Path(cmd[cmd.index("--result") + 1])
            captured["payload"] = json.loads(request_path.read_text(encoding="utf-8"))
            result_path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "video_path": str(tmp_path / "video.mp4"),
                        "poster_path": str(tmp_path / "poster.jpg"),
                        "seed": captured["payload"]["seed"],
                        "frame_count": captured["payload"]["num_frames"],
                        "fps": captured["payload"]["fps"],
                    }
                ),
                encoding="utf-8",
            )
            self.returncode = 0

        def poll(self):
            return self.returncode

    return FakeProcess


def test_wan_i2v_backend_serializes_source_image_and_preserves_zero_seed(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        "wan_backend.subprocess.Popen",
        _fake_process_factory(captured, tmp_path),
    )
    descriptor = describe_video_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    backend = WanDiffusersBackend()
    result = backend.generate(
        descriptor,
        {
            "prompt": "camera moves forward",
            "image_path": "/tmp/source.png",
            "seed": 0,
        },
        output_dir=tmp_path,
    )

    assert result["ok"] is True
    assert backend.can_handle(descriptor) is True
    assert captured["payload"]["input_image"] == "/tmp/source.png"
    assert captured["payload"]["seed"] == 0
    assert captured["payload"]["num_frames"] == 81


def test_wan_i2v_rejects_missing_required_source(tmp_path):
    descriptor = describe_video_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    with pytest.raises(ValueError, match="requires a source image"):
        WanDiffusersBackend().generate(
            descriptor,
            {"prompt": "move"},
            output_dir=tmp_path,
        )


def test_wan_t2v_and_ti2v_are_owned_by_same_backend(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        "wan_backend.subprocess.Popen",
        _fake_process_factory(captured, tmp_path),
    )

    t2v = describe_video_model("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    assert t2v.capabilities.text_to_video is True
    assert WanDiffusersBackend().can_handle(t2v) is True
    WanDiffusersBackend().generate(
        t2v,
        {"prompt": "a duck crosses a rainy street", "seed": 7},
        output_dir=tmp_path,
    )
    assert captured["payload"]["input_image"] is None

    ti2v = describe_video_model("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    assert ti2v.capabilities.text_to_video is True
    assert ti2v.capabilities.image_to_video is True
    assert WanDiffusersBackend().can_handle(ti2v) is True


def test_wan_worker_snaps_dimensions_and_frames():
    assert _snap_dimension(839) == 832
    assert _snap_dimension(480) == 480
    assert _snap_frames(80) == 77
    assert _snap_frames(81) == 81
    assert _snap_frames(84) == 81


def test_wan_backend_is_process_isolated():
    import inspect

    source = inspect.getsource(WanDiffusersBackend)
    assert "subprocess.Popen" in source
    assert "_generate_frames_with_diffusers" not in source
    assert "_prepare_runtime_for_wan" not in source
    assert "implementation" not in source.lower()
