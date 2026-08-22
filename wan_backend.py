"""Wan backend behind DuckMotion's generic VideoBackend contract."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from model_runtime import VideoBackend, VideoModelDescriptor, backend_resolver
from runtime_probe import probe_python_runtime


def _gguf_pair_mate(path: Path) -> Path | None:
    stem = path.stem
    match = re.search(r"(?i)^(.*?)([_ .-]?)([hl])$", stem)
    if match:
        other = "L" if match.group(3).upper() == "H" else "H"
        candidate = path.with_name(f"{match.group(1)}{match.group(2)}{other}{path.suffix}")
        if candidate.exists():
            return candidate
    return None


class WanDiffusersBackend(VideoBackend):
    backend_id = "wan_diffusers"
    _readiness_ttl_seconds = 60.0

    def __init__(self) -> None:
        self._readiness_checked_at = 0.0
        self._readiness_key: tuple[str, str, str] | None = None
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
        detection = descriptor.detection or {}
        source_format = str(detection.get("format") or "diffusers").lower()
        pair_role = str(detection.get("pair_role") or "").upper()
        pair_state = "none"
        if source_format == "gguf" and pair_role in {"H", "L"}:
            source_path = Path(descriptor.source).expanduser()
            mate = _gguf_pair_mate(source_path)
            if mate is None:
                return {
                    "ready": False,
                    "source_format": "gguf",
                    "reason": (
                        f"Wan GGUF checkpoint '{source_path.name}' is an {pair_role} half, "
                        "but its matching H/L GGUF file is missing. Put both files beside each "
                        "other; DuckMotion will not download a stock second transformer as a fallback."
                    ),
                }
            pair_state = str(mate)

        readiness_key = (python_exe, source_format, pair_state)
        now = time.monotonic()
        if (
            self._readiness_payload is not None
            and self._readiness_key == readiness_key
            and now - self._readiness_checked_at < self._readiness_ttl_seconds
        ):
            return dict(self._readiness_payload)

        symbols: list[tuple[str, str]] = [
            ("diffusers", "WanPipeline"),
            ("diffusers", "WanImageToVideoPipeline"),
            ("diffusers.utils", "export_to_video"),
            ("diffusers.utils", "load_image"),
        ]
        if source_format == "gguf":
            symbols.extend(
                [
                    ("diffusers", "WanTransformer3DModel"),
                    ("diffusers", "GGUFQuantizationConfig"),
                    ("gguf", "GGUFReader"),
                ]
            )

        payload = probe_python_runtime(python_exe, tuple(symbols))
        payload["source_format"] = source_format
        if pair_state != "none":
            payload["gguf_pair_present"] = True
        self._readiness_checked_at = now
        self._readiness_key = readiness_key
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
            "model_name": descriptor.name,
            "source_format": str((descriptor.detection or {}).get("format") or "diffusers").lower(),
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
        return None


_backend = WanDiffusersBackend()


def ensure_registered() -> WanDiffusersBackend:
    if _backend.backend_id not in backend_resolver.ids():
        backend_resolver.register(_backend)
    return _backend
