"""Isolated LTX-2.5 runtime adapter for DuckMotion."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import host_runtime
from pathlib import Path
from typing import Any, Callable

from model_runtime import VideoBackend, VideoModelDescriptor, backend_resolver
from runtime_probe import probe_python_runtime


class LTX25IsolatedBackend(VideoBackend):
    backend_id = "ltx25_isolated"
    _readiness_ttl_seconds = 60.0

    def __init__(self) -> None:
        self._readiness_checked_at = 0.0
        self._readiness_python = ""
        self._readiness_payload: dict[str, Any] | None = None

    def can_handle(self, descriptor: VideoModelDescriptor) -> bool:
        return (
            descriptor.backend == self.backend_id
            and descriptor.architecture == "ltx25"
            and (descriptor.capabilities.text_to_video or descriptor.capabilities.image_to_video)
        )

    def readiness(self, descriptor: VideoModelDescriptor) -> dict[str, Any]:
        if not self.can_handle(descriptor):
            return {
                "ready": False,
                "reason": f"The installed runtime does not implement this workflow for '{descriptor.name}'.",
            }

        python_exe = os.getenv("DUCKMOTION_LTX_PYTHON") or sys.executable
        now = time.monotonic()
        if (
            self._readiness_payload is not None
            and self._readiness_python == python_exe
            and now - self._readiness_checked_at < self._readiness_ttl_seconds
        ):
            return dict(self._readiness_payload)

        payload = probe_python_runtime(
            python_exe,
            (
                ("diffusers", "LTX2Pipeline"),
                ("diffusers", "LTX2ImageToVideoPipeline"),
                ("diffusers", "LTX2LatentUpsamplePipeline"),
                ("diffusers.pipelines.ltx2.latent_upsampler", "LTX2LatentUpsamplerModel"),
                ("diffusers.pipelines.ltx2.utils", "DISTILLED_SIGMA_VALUES"),
                ("diffusers.pipelines.ltx2.utils", "STAGE_2_DISTILLED_SIGMA_VALUES"),
                ("diffusers.utils", "encode_video"),
            ),
        )
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
            raise ValueError("LTX output_dir is required")
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
            "input_image": str(request.get("image_path") or "").strip() or None,
            "width": int(request.get("width") or defaults.get("width") or 1536),
            "height": int(request.get("height") or defaults.get("height") or 1024),
            "num_frames": int(request.get("num_frames") or defaults.get("num_frames") or 121),
            "fps": float(request.get("fps") or defaults.get("fps") or 24),
            "seed": seed,
        }
        if not payload["prompt"]:
            raise ValueError("Prompt is required")

        python_exe = os.getenv("DUCKMOTION_LTX_PYTHON") or sys.executable
        worker = Path(__file__).with_name("ltx_worker.py")
        timeout_seconds = max(60.0, float(os.getenv("DUCKMOTION_LTX_TIMEOUT_SECONDS", "7200")))

        with tempfile.TemporaryDirectory(prefix="duckmotion_ltx_") as tmp_raw:
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
                heartbeat_tick = 0
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
                        raise RuntimeError(
                            f"LTX runtime timed out after {int(timeout_seconds)} seconds"
                        )
                    heartbeat_tick += 1
                    if heartbeat_tick % 10 == 0:
                        try:
                            host_runtime.lease_heartbeat(token=kwargs.get("lease_token") or "")
                        except Exception:
                            pass
                    time.sleep(0.5)

            logs = (
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                if log_path.exists()
                else []
            )
            if not result_path.exists():
                raise RuntimeError(
                    f"LTX runtime exited without a result (code {proc.returncode}).\n"
                    + "\n".join(logs[-20:])
                )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("ok"):
                raise RuntimeError(
                    (
                        str(result.get("error") or "LTX runtime failed")
                        + "\n"
                        + "\n".join(logs[-20:])
                    ).strip()
                )
            return result

    def unload(self) -> None:
        return None


_backend = LTX25IsolatedBackend()


def ensure_registered() -> LTX25IsolatedBackend:
    if _backend.backend_id not in backend_resolver.ids():
        backend_resolver.register(_backend)
    return _backend
