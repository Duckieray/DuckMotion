"""Wan backend behind DuckMotion's generic VideoBackend contract."""

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


class WanDiffusersBackend(VideoBackend):
    backend_id = "wan_diffusers"
    _readiness_ttl_seconds = 60.0

    def __init__(self) -> None:
        self._readiness_checked_at = 0.0
        self._readiness_python = ""
        self._readiness_payload: dict[str, Any] | None = None

    def can_handle(self, descriptor: VideoModelDescriptor) -> bool:
        return (
            descriptor.backend == self.backend_id
            and descriptor.architecture == "wan22"
            and (descriptor.capabilities.text_to_video or descriptor.capabilities.image_to_video)
        )

    def readiness(self, descriptor: VideoModelDescriptor) -> dict[str, Any]:
        if not self.can_handle(descriptor):
            return {
                "ready": False,
                "reason": f"The installed runtime does not implement this workflow for '{descriptor.name}'.",
            }

        python_exe = os.getenv("DUCKMOTION_WAN_PYTHON") or sys.executable
        now = time.monotonic()
        if (
            self._readiness_payload is not None
            and self._readiness_python == python_exe
            and now - self._readiness_checked_at < self._readiness_ttl_seconds
        ):
            return dict(self._readiness_payload)

        probe = (
            "import torch; "
            "from diffusers import WanImageToVideoPipeline, WanPipeline; "
            "from diffusers.utils import export_to_video, load_image"
        )
        try:
            completed = subprocess.run(
                [python_exe, "-c", probe],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if completed.returncode == 0:
                payload = {"ready": True, "reason": None}
            else:
                detail = (completed.stderr or completed.stdout or "").strip().splitlines()
                reason = detail[-1] if detail else f"Runtime probe exited with code {completed.returncode}."
                payload = {"ready": False, "reason": f"Wan runtime dependencies unavailable: {reason}"}
        except Exception as exc:
            payload = {"ready": False, "reason": f"Unable to probe Wan runtime: {exc}"}

        self._readiness_checked_at = now
        self._readiness_python = python_exe
        self._readiness_payload = dict(payload)
        return payload

    def generate(
        self,
        descriptor: VideoModelDescriptor,
        request: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        output_dir_raw = str(kwargs.get("output_dir") or "").strip()
        if not output_dir_raw:
            raise ValueError("Wan output_dir is required")
        output_dir = Path(output_dir_raw).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        is_cancelled: Callable[[], bool] | None = kwargs.get("is_cancelled")
        defaults = descriptor.defaults or {}
        seed = int(
            request.get("seed")
            if request.get("seed") is not None
            else int(time.time_ns() & 0xFFFFFFFF)
        )
        payload = {
            "model_path": descriptor.source,
            "prompt": str(request.get("prompt") or "").strip(),
            "negative_prompt": str(request.get("negative_prompt") or ""),
            "input_image": str(request.get("image_path") or "").strip() or None,
            "width": int(request.get("width") or defaults.get("width") or 832),
            "height": int(request.get("height") or defaults.get("height") or 480),
            "num_frames": int(request.get("num_frames") or defaults.get("num_frames") or 81),
            "fps": int(request.get("fps") or defaults.get("fps") or 16),
            "num_inference_steps": int(
                request.get("num_inference_steps")
                or defaults.get("num_inference_steps")
                or 30
            ),
            "guidance_scale": float(
                request.get("guidance_scale")
                if request.get("guidance_scale") is not None
                else defaults.get("guidance_scale", 5.0)
            ),
            "seed": seed,
        }
        if not payload["prompt"]:
            raise ValueError("Prompt is required")
        if descriptor.capabilities.source_image_required and not payload["input_image"]:
            raise ValueError("The selected Wan model requires a source image")
        if payload["input_image"] and not descriptor.capabilities.image_to_video:
            raise ValueError("The selected Wan model does not support image conditioning")

        python_exe = os.getenv("DUCKMOTION_WAN_PYTHON") or sys.executable
        worker = Path(__file__).with_name("wan_worker.py")
        timeout_seconds = max(60.0, float(os.getenv("DUCKMOTION_WAN_TIMEOUT_SECONDS", "7200")))

        with tempfile.TemporaryDirectory(prefix="duckmotion_wan_") as tmp_raw:
            tmp = Path(tmp_raw)
            request_path = tmp / "request.json"
            result_path = tmp / "result.json"
            log_path = tmp / "worker.log"
            request_path.write_text(json.dumps(payload), encoding="utf-8")

            with log_path.open("w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(
                    [
                        python_exe,
                        str(worker),
                        "--request",
                        str(request_path),
                        "--result",
                        str(result_path),
                        "--output-dir",
                        str(output_dir),
                    ],
                    stdout=log_file,
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
                        raise RuntimeError("Wan generation cancelled")
                    if time.monotonic() - started > timeout_seconds:
                        proc.kill()
                        raise RuntimeError(
                            f"Wan runtime timed out after {int(timeout_seconds)} seconds"
                        )
                    time.sleep(0.5)

            logs = (
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                if log_path.exists()
                else []
            )
            if not result_path.exists():
                raise RuntimeError(
                    f"Wan runtime exited without a result (code {proc.returncode}).\n"
                    + "\n".join(logs[-20:])
                )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("ok"):
                raise RuntimeError(
                    (
                        str(result.get("error") or "Wan runtime failed")
                        + "\n"
                        + "\n".join(logs[-20:])
                    ).strip()
                )
            return result

    def unload(self) -> None:
        # Wan is process-isolated; worker exit releases its model resources.
        return None


_backend = WanDiffusersBackend()


def ensure_registered() -> WanDiffusersBackend:
    if _backend.backend_id not in backend_resolver.ids():
        backend_resolver.register(_backend)
    return _backend
