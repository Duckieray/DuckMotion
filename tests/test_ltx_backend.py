from __future__ import annotations

import json
from pathlib import Path

from ltx_backend import LTX25IsolatedBackend
from model_runtime import describe_video_model


def test_ltx_backend_serializes_distilled_request(monkeypatch, tmp_path):
    captured = {}

    class FakeStdout:
        def read(self):
            return ""

    class FakeProcess:
        def __init__(self, cmd, **_kwargs):
            captured["cmd"] = cmd
            request_path = Path(cmd[cmd.index("--request") + 1])
            result_path = Path(cmd[cmd.index("--result") + 1])
            captured["payload"] = json.loads(request_path.read_text(encoding="utf-8"))
            result_path.write_text(
                json.dumps({
                    "ok": True,
                    "video_path": str(tmp_path / "video.mp4"),
                    "poster_path": str(tmp_path / "poster.jpg"),
                    "seed": 42,
                    "frame_count": 121,
                    "fps": 24,
                }),
                encoding="utf-8",
            )
            self.returncode = 0
            self.stdout = FakeStdout()

        def poll(self):
            return self.returncode

    monkeypatch.setattr("ltx_backend.subprocess.Popen", FakeProcess)
    descriptor = describe_video_model("Lightricks/LTX-2.5-Diffusers")
    backend = LTX25IsolatedBackend()
    result = backend.generate(
        descriptor,
        {
            "prompt": "a red fox walking through snow",
            "width": 768,
            "height": 512,
            "num_frames": 121,
            "fps": 24,
            "seed": 42,
        },
        output_dir=tmp_path,
    )

    assert result["ok"] is True
    assert captured["payload"]["model_path"] == "Lightricks/LTX-2.5-Diffusers"
    assert captured["payload"]["input_image"] is None
    assert captured["payload"]["num_frames"] == 121
    assert captured["payload"]["seed"] == 42


def test_ltx_backend_can_receive_image_conditioning(monkeypatch, tmp_path):
    captured = {}

    class FakeStdout:
        def read(self):
            return ""

    class FakeProcess:
        def __init__(self, cmd, **_kwargs):
            request_path = Path(cmd[cmd.index("--request") + 1])
            result_path = Path(cmd[cmd.index("--result") + 1])
            captured.update(json.loads(request_path.read_text(encoding="utf-8")))
            result_path.write_text(json.dumps({"ok": True, "seed": 1}), encoding="utf-8")
            self.returncode = 0
            self.stdout = FakeStdout()

        def poll(self):
            return self.returncode

    monkeypatch.setattr("ltx_backend.subprocess.Popen", FakeProcess)
    descriptor = describe_video_model("Lightricks/LTX-2.5-Diffusers")
    LTX25IsolatedBackend().generate(
        descriptor,
        {"prompt": "camera pushes in", "image_path": "/tmp/source.png", "seed": 1},
        output_dir=tmp_path,
    )
    assert captured["input_image"] == "/tmp/source.png"
