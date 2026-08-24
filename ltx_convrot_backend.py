"""LTX-2.5 INT8 ConvRot backend behind the generic video model contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Callable

from ltx_convrot_assets import inspect_convrot_assets, read_json
from ltx_convrot_quality import apply_quality_asset_policy
from ltx_convrot_recipe import extract_execution_recipe
from model_recipes import ExecutionProfile, execution_profiles
from model_runtime import VideoBackend, VideoModelDescriptor, backend_resolver


class LTX25ConvRotBackend(VideoBackend):
    backend_id = "ltx25_convrot"
    _readiness_ttl_seconds = 60.0

    def __init__(self) -> None:
        self._readiness_key: tuple[str, str, str, str] | None = None
        self._readiness_checked_at = 0.0
        self._readiness_payload: dict[str, Any] | None = None

    def can_handle(self, descriptor: VideoModelDescriptor) -> bool:
        return (
            descriptor.backend == self.backend_id
            and descriptor.architecture == "ltx25"
            and str((descriptor.detection or {}).get("format") or "") == "int8_convrot"
        )

    @staticmethod
    def _python() -> str:
        return str(os.getenv("DUCKMOTION_LTX_CONVROT_PYTHON") or "").strip()

    @staticmethod
    def _comfy_root(python_exe: str) -> Path:
        override = str(os.getenv("DUCKMOTION_LTX_CONVROT_COMFY_ROOT") or "").strip()
        if override:
            return Path(override).expanduser()
        if not python_exe:
            return Path("<missing>")
        return Path(python_exe).expanduser().resolve().parent.parent / "comfyui"

    @staticmethod
    def _profile(assets: dict[str, Any]) -> ExecutionProfile | None:
        return execution_profiles.get(str(assets.get("execution_profile") or ""))

    @staticmethod
    def _asset_state(
        checkpoint: str,
        *,
        models_dir: str | None = None,
    ) -> dict[str, Any]:
        state = inspect_convrot_assets(checkpoint, models_dir=models_dir)
        return apply_quality_asset_policy(state, checkpoint, models_dir=models_dir)

    @staticmethod
    def _execution_recipe(assets: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(assets.get("config_path") or "").strip()
        config = read_json(Path(raw_path)) if raw_path else {}
        return extract_execution_recipe(config)

    def _probe_runtime(
        self,
        python_exe: str,
        comfy_root: Path,
        profile: ExecutionProfile | None,
    ) -> dict[str, Any]:
        executable = Path(python_exe).expanduser() if python_exe else Path("<missing>")
        if not python_exe or not executable.exists():
            return {
                "ready": False,
                "python": python_exe,
                "comfy_root": str(comfy_root),
                "reason": "LTX ConvRot runtime Python is not configured or does not exist.",
            }
        if not comfy_root.exists():
            return {
                "ready": False,
                "python": python_exe,
                "comfy_root": str(comfy_root),
                "reason": "Pinned Comfy core checkout is missing from the ConvRot runtime.",
            }

        required_nodes = sorted(profile.required_runtime_nodes) if profile else []
        script = r'''
import asyncio, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
required_nodes = set(json.loads(sys.argv[2]))
sys.path.insert(0, str(root))
sys.argv = [sys.argv[0]]
out = {"ready": True, "comfy_root": str(root), "missing": [], "missing_nodes": []}
try:
    import torch
    out["torch_version"] = getattr(torch, "__version__", None)
    out["cuda_available"] = bool(torch.cuda.is_available())
    if out["cuda_available"]:
        out["gpu_name"] = torch.cuda.get_device_name(0)
        out["cuda_capability"] = list(torch.cuda.get_device_capability(0))
        out["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
except Exception as exc:
    out["ready"] = False; out["missing"].append({"module": "torch", "error": str(exc)})
for module_name, symbol in (("comfy.sd", "load_diffusion_model"), ("comfy.sd", "load_clip"), ("comfy_kitchen", None), ("av", None)):
    try:
        module = __import__(module_name, fromlist=["*"])
        if symbol: getattr(module, symbol)
    except Exception as exc:
        out["ready"] = False; out["missing"].append({"module": module_name, "symbol": symbol, "error": str(exc)})
try:
    import nodes
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    out["missing_nodes"] = sorted(required_nodes.difference(nodes.NODE_CLASS_MAPPINGS))
    if out["missing_nodes"]:
        out["ready"] = False
except Exception as exc:
    out["ready"] = False; out["missing"].append({"module": "nodes", "error": str(exc)})
if out["ready"] and not out.get("cuda_available"):
    out["ready"] = False; out["reason"] = "ConvRot runtime imports succeed, but CUDA is unavailable."
elif out["missing_nodes"]:
    out["reason"] = "Pinned Comfy core is missing nodes required by the selected execution profile."
elif out["missing"]:
    out["reason"] = "One or more ConvRot runtime imports are unavailable."
print(json.dumps(out))
'''
        try:
            completed = subprocess.run(
                [python_exe, "-c", script, str(comfy_root), json.dumps(required_nodes)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except Exception as exc:
            return {"ready": False, "python": python_exe, "reason": f"ConvRot probe failed to start: {exc}"}
        lines = (completed.stdout or "").strip().splitlines()
        if not lines:
            return {
                "ready": False,
                "python": python_exe,
                "comfy_root": str(comfy_root),
                "reason": f"ConvRot probe returned no JSON (exit {completed.returncode}).",
                "detail": (completed.stderr or "")[-4000:],
            }
        try:
            payload = json.loads(lines[-1])
        except Exception:
            return {
                "ready": False,
                "python": python_exe,
                "comfy_root": str(comfy_root),
                "reason": "ConvRot probe returned invalid JSON.",
                "detail": "\n".join(lines[-10:])[-4000:],
            }
        payload["python"] = python_exe
        payload["exit_code"] = completed.returncode
        if completed.returncode != 0:
            payload["ready"] = False
            payload.setdefault("reason", "ConvRot runtime probe exited unsuccessfully.")
        return payload

    def readiness(self, descriptor: VideoModelDescriptor) -> dict[str, Any]:
        if not self.can_handle(descriptor):
            return {"ready": False, "reason": "This backend does not own the selected model."}

        python_exe = self._python()
        comfy_root = self._comfy_root(python_exe)
        assets = self._asset_state(descriptor.source)
        profile = self._profile(assets)
        profile_id = profile.profile_id if profile else ""
        key = (python_exe, str(comfy_root), descriptor.source, profile_id)
        now = time.monotonic()
        if (
            self._readiness_payload is not None
            and self._readiness_key == key
            and now - self._readiness_checked_at < self._readiness_ttl_seconds
        ):
            return dict(self._readiness_payload)

        runtime = self._probe_runtime(python_exe, comfy_root, profile)
        ready = bool(runtime.get("ready") and profile is not None and assets.get("ready"))
        reason = runtime.get("reason")
        if runtime.get("ready") and profile is None:
            reason = "LTX ConvRot format is supported, but no compatible execution recipe was resolved."
        elif runtime.get("ready") and not assets.get("ready"):
            reason = "LTX ConvRot checkpoint recipe/assets are incomplete: " + ", ".join(assets.get("missing") or [])
        payload = {
            **runtime,
            "ready": ready,
            "reason": reason,
            "source_format": "int8_convrot",
            "execution_profile": profile_id or None,
            "execution_recipe": self._execution_recipe(assets) if profile else None,
            "profile_defaults": dict(profile.defaults) if profile else {},
            "profile_constraints": dict(profile.constraints) if profile else {},
            "assets": assets,
        }
        self._readiness_key = key
        self._readiness_checked_at = now
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
            raise ValueError("LTX ConvRot output_dir is required")
        output_dir = Path(output_dir_raw).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        config = kwargs.get("config") if isinstance(kwargs.get("config"), dict) else {}
        models_dir = str(config.get("models_dir") or "") or None
        assets = self._asset_state(descriptor.source, models_dir=models_dir)
        if not assets.get("ready"):
            raise RuntimeError(
                "LTX ConvRot recipe/assets are incomplete: "
                + ", ".join(assets.get("missing") or [])
            )
        profile = self._profile(assets)
        if profile is None:
            raise RuntimeError("No compatible LTX ConvRot execution profile is installed")

        python_exe = self._python()
        if not python_exe:
            raise RuntimeError("LTX ConvRot runtime Python is not configured")
        comfy_root = self._comfy_root(python_exe)
        worker = Path(__file__).with_name(profile.worker)
        if not worker.exists():
            raise RuntimeError(
                f"Execution profile '{profile.profile_id}' worker is missing: {worker}"
            )
        is_cancelled: Callable[[], bool] | None = kwargs.get("is_cancelled")
        defaults = dict(descriptor.defaults or {})
        defaults.update(dict(profile.defaults))
        seed = int(
            request.get("seed")
            if request.get("seed") is not None
            else int(time.time_ns() & 0xFFFFFFFF)
        )
        payload = {
            "model_path": descriptor.source,
            "model_name": descriptor.name,
            "execution_profile": profile.profile_id,
            "execution_recipe": self._execution_recipe(assets),
            "config_path": assets.get("config_path"),
            "asset_policy": assets.get("asset_policy"),
            "quality_upgrades": assets.get("quality_upgrades") or {},
            "assets": assets.get("assets"),
            "prompt": str(request.get("prompt") or "").strip(),
            "input_image": str(request.get("image_path") or "").strip() or None,
            "i2v_stability": str(request.get("i2v_stability") or "").strip() or None,
            "width": int(request.get("width") or defaults.get("width") or 1152),
            "height": int(request.get("height") or defaults.get("height") or 768),
            "num_frames": int(request.get("num_frames") or defaults.get("num_frames") or 241),
            "fps": int(request.get("fps") or defaults.get("fps") or 24),
            "seed": seed,
        }
        if not payload["prompt"]:
            raise ValueError("Prompt is required")

        timeout_seconds = max(
            60.0,
            float(os.getenv("DUCKMOTION_LTX_CONVROT_TIMEOUT_SECONDS", "14400")),
        )
        with tempfile.TemporaryDirectory(prefix="duckmotion_ltx_convrot_") as tmp_raw:
            tmp = Path(tmp_raw)
            request_path = tmp / "request.json"
            result_path = tmp / "result.json"
            log_path = tmp / "worker.log"
            request_path.write_text(json.dumps(payload), encoding="utf-8")
            env = dict(os.environ)
            env["DUCKMOTION_LTX_CONVROT_COMFY_ROOT"] = str(comfy_root)
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
                    env=env,
                )
                started = time.monotonic()
                while proc.poll() is None:
                    if is_cancelled is not None and is_cancelled():
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise RuntimeError("LTX ConvRot generation cancelled")
                    if time.monotonic() - started > timeout_seconds:
                        proc.kill()
                        raise RuntimeError(
                            f"LTX ConvRot runtime timed out after {int(timeout_seconds)} seconds"
                        )
                    time.sleep(0.5)

            logs = (
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                if log_path.exists()
                else []
            )
            if not result_path.exists():
                raise RuntimeError(
                    f"LTX ConvRot runtime exited without a result (code {proc.returncode}).\n"
                    + "\n".join(logs[-20:])
                )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("ok"):
                worker_trace = str(result.get("traceback") or "").strip()
                detail = str(result.get("error") or "LTX ConvRot runtime failed")
                if worker_trace:
                    detail = f"{detail}\n{worker_trace}"
                raise RuntimeError((detail + "\n" + "\n".join(logs[-20:])).strip())
            return result

    def unload(self) -> None:
        return None


_backend = LTX25ConvRotBackend()


def ensure_registered() -> LTX25ConvRotBackend:
    if _backend.backend_id not in backend_resolver.ids():
        backend_resolver.register(_backend)
    return _backend
