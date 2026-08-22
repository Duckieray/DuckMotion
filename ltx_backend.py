"""Isolated LTX-2.5 runtime adapter for DuckMotion."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from model_runtime import VideoBackend, VideoModelDescriptor, backend_resolver


class LTX25IsolatedBackend(VideoBackend):
    backend_id = "ltx25_isolated"

    def can_handle(self, descriptor: VideoModelDescriptor) -> bool:
        return descriptor.backend == self.backend_id and descriptor.architecture == "ltx25"

    def generate(
        self,
        descriptor: VideoModelDescriptor,
        request: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        output_dir = Path(str(kwargs.get("output_dir") or "")).expanduser()
        if not str(output_dir):
            raise ValueError("LTX output_dir is required")
        output_dir.mkdir(parents=True, exist_ok=True)

        is_cancelled: Callable[[], bool] | None = kwargs.get("is_cancelled")
        defaults = descriptor.defaults or {}
        seed = int(request.get("seed") if request.get("seed") is not None else int(time.time_ns() & 0xFFFFFFFF))
        payload = {
            "model_path": descriptor.source,
            "prompt": str(request.get("prompt") or "").strip(),
            "input_image": str(request.get("image_path") or "").strip() or None,
            "width": int(request.get("width") or defaults.get("width") or 768),
            "height": int(request.get("height") or defaults.get("height") or 512),
            "num_frames": int(request.get("num_frames") or defaults.get("num_frames") or 121),
            "fps": float(request.get("fps") or defaults.get("fps") or 24),
            "seed": seed,
        }
        if not payload["prompt"]:
            raise ValueError("Prompt is required")

        python_exe = os.getenv("DUCKMOTION_LTX_PYTHON") or sys.executable
        worker = Path(__file__).with_name("ltx_worker.py")
        timeout_seconds = max(60.0, float(os.getenv("DUCKMOTION_LTX_TIMEOUT_SECONDS", "3600")))

        with tempfile.TemporaryDirectory(prefix="duckmotion_ltx_") as tmp_raw:
            tmp = Path(tmp_raw)
            request_path = tmp / "request.json"
            result_path = tmp / "result.json"
            request_path.write_text(json.dumps(payload), encoding="utf-8")

            proc = subprocess.Popen(
                [python_exe, str(worker), "--request", str(request_path), "--result", str(result_path), "--output-dir", str(output_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            started = time.monotonic()
            while proc.poll() is None:
                if is_cancelled is not None and is_cancelled():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise RuntimeError("LTX generation cancelled")
                if time.monotonic() - started > timeout_seconds:
                    proc.kill()
                    raise RuntimeError(f"LTX runtime timed out after {int(timeout_seconds)} seconds")
                time.sleep(0.5)

            logs = proc.stdout.read().splitlines()[-80:] if proc.stdout is not None else []
            if not result_path.exists():
                raise RuntimeError(
                    f"LTX runtime exited without a result (code {proc.returncode}).\n" + "\n".join(logs[-20:])
                )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("ok"):
                raise RuntimeError(
                    (str(result.get("error") or "LTX runtime failed") + "\n" + "\n".join(logs[-20:])).strip()
                )
            return result


_backend = LTX25IsolatedBackend()


def ensure_registered() -> LTX25IsolatedBackend:
    if _backend.backend_id not in backend_resolver.ids():
        backend_resolver.register(_backend)
    return _backend
