from __future__ import annotations

import inspect
import json
from pathlib import Path

from ltx_backend import LTX25IsolatedBackend
from ltx_worker import _snap_final_dimension, _snap_frames
from model_runtime import describe_video_model


def _fake_process_factory(captured, tmp_path):
    class FakeProcess:
        def __init__(self, cmd, **_kwargs):
            request_path = Path(cmd[cmd.index("--request") + 1])
            result_path = Path(cmd[cmd.index("--result") + 1])
            captured.update(json.loads(request_path.read_text(encoding="utf-8")))
            result_path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "video_path": str(tmp_path / "video.mp4"),
                        "poster_path": str(tmp_path / "poster.jpg"),
                        "seed": captured["seed"],
                        "frame_count": captured["num_frames"],
                        "fps": captured["fps"],
                    }
                ),
                encoding="utf-8",
            )
            self.returncode = 0

        def poll(self):
            return self.returncode

    return FakeProcess


def test_ltx_backend_serializes_final_resolution_request(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        "ltx_backend.subprocess.Popen",
        _fake_process_factory(captured, tmp_path),
    )
    descriptor = describe_video_model("Lightricks/LTX-2.5-Diffusers")
    backend = LTX25IsolatedBackend()
    result = backend.generate(
        descriptor,
        {
            "prompt": "a red fox walking through snow",
            "width": 1536,
            "height": 1024,
            "num_frames": 121,
            "fps": 24,
            "seed": 0,
        },
        output_dir=tmp_path,
    )

    assert result["ok"] is True
    assert captured["model_path"] == "Lightricks/LTX-2.5-Diffusers"
    assert captured["input_image"] is None
    assert captured["width"] == 1536
    assert captured["height"] == 1024
    assert captured["num_frames"] == 121
    assert captured["seed"] == 0


def test_ltx_backend_can_receive_image_conditioning(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        "ltx_backend.subprocess.Popen",
        _fake_process_factory(captured, tmp_path),
    )
    descriptor = describe_video_model("Lightricks/LTX-2.5-Diffusers")
    LTX25IsolatedBackend().generate(
        descriptor,
        {"prompt": "camera pushes in", "image_path": "/tmp/source.png", "seed": 1},
        output_dir=tmp_path,
    )
    assert captured["input_image"] == "/tmp/source.png"


def test_ltx_worker_enforces_two_stage_dimension_and_frame_constraints():
    assert _snap_final_dimension(1555) == 1536
    assert _snap_final_dimension(1050) == 1024
    assert _snap_final_dimension(1536) % 64 == 0
    assert _snap_frames(120) == 113
    assert _snap_frames(121) == 121


def test_ltx_worker_source_contains_reference_two_stage_primitives():
    import ltx_worker

    source = inspect.getsource(ltx_worker._run)
    assert "LTX2LatentUpsamplerModel" in source
    assert "LTX2LatentUpsamplePipeline" in source
    assert "DISTILLED_SIGMA_VALUES" in source
    assert "STAGE_2_DISTILLED_SIGMA_VALUES" in source
    assert 'output_type="latent"' in source
    assert "audio_latents=audio_latent" in source
    assert "noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0]" in source
