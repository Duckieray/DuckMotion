"""Architecture-neutral DuckMotion health, config, and status surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from model_runtime import VideoBackendResolver, VideoModelDescriptor


class VideoConfigPayload(BaseModel):
    model_id_or_path: str | None = None
    models_dir: str | None = None
    output_dir: str | None = None


def _clean_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return str(Path(raw).expanduser())


class VideoRuntimeSurfaces:
    def __init__(
        self,
        *,
        services: Any,
        resolver: VideoBackendResolver,
        describe_model: Callable[[str], VideoModelDescriptor],
        discover_models: Callable[[dict[str, Any]], dict[str, Any]],
        register_backends: Callable[[], None],
    ) -> None:
        self.services = services
        self.resolver = resolver
        self.describe_model = describe_model
        self.discover_models = discover_models
        self.register_backends = register_backends

    def public_config(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "model_id_or_path": str(config.get("model_id_or_path") or "").strip(),
            "models_dir": _clean_path(config.get("models_dir")),
            "output_dir": _clean_path(config.get("output_dir")),
            "resolved_output_dir": str(self.services.resolve_output_dir(config)),
        }

    def get_config(self) -> dict[str, Any]:
        return {"config": self.public_config(self.services.load_config())}

    def set_config(self, payload: VideoConfigPayload) -> dict[str, Any]:
        current = self.services.load_config()
        next_config = dict(current)
        if payload.model_id_or_path is not None:
            next_config["model_id_or_path"] = str(payload.model_id_or_path or "").strip()
        if payload.models_dir is not None:
            next_config["models_dir"] = _clean_path(payload.models_dir)
        if payload.output_dir is not None:
            next_config["output_dir"] = _clean_path(payload.output_dir)
        self.services.save_config(next_config)
        return {"ok": True, "config": self.public_config(next_config)}

    def _selected_model(self, config: dict[str, Any]) -> VideoModelDescriptor | None:
        source = str(config.get("model_id_or_path") or "").strip()
        return self.describe_model(source) if source else None

    def _model_runnable(self, descriptor: VideoModelDescriptor | None) -> tuple[bool, str | None]:
        if descriptor is None:
            return False, "Select a video model."
        if not descriptor.supported:
            return False, f"No runnable backend is installed for video model '{descriptor.name}'."
        try:
            self.register_backends()
            readiness = self.resolver.readiness(descriptor)
        except Exception as exc:
            return False, str(exc)
        if not readiness.get("ready"):
            return False, str(readiness.get("reason") or "Selected model runtime is not ready.")
        return True, None

    def _snapshot(self) -> dict[str, Any]:
        config = self.services.load_config()
        descriptor = self._selected_model(config)
        runtime = self.services.runtime_profile_safe()
        output_ok, output_error = self.services.output_writable(config)
        model_ok, model_error = self._model_runnable(descriptor)
        runtime_ok = bool(runtime.get("ok"))
        reasons = [
            reason
            for reason in (
                model_error,
                None if runtime_ok else str(runtime.get("error") or "Runtime profile unavailable."),
                None if output_ok else str(output_error or "Output directory is not writable."),
            )
            if reason
        ]
        return {
            "config": config,
            "model": descriptor,
            "runtime": runtime,
            "output_ok": bool(output_ok),
            "output_error": output_error,
            "model_ok": bool(model_ok),
            "runtime_ok": runtime_ok,
            "reasons": reasons,
            "engine_ready": bool(model_ok and runtime_ok and output_ok),
        }

    def health(self) -> dict[str, Any]:
        state = self._snapshot()
        catalog = self.discover_models(state["config"])
        descriptor = state["model"]
        return {
            "ok": True,
            "mode": "model-driven-local-runtime",
            "config": self.public_config(state["config"]),
            "selected_model": descriptor.to_public_dict() if descriptor is not None else None,
            "runtime": state["runtime"],
            "ready": {
                "model_selected": descriptor is not None,
                "model_runnable": state["model_ok"],
                "runtime_ready": state["runtime_ok"],
                "output_ready": state["output_ok"],
                "engine_ready": state["engine_ready"],
            },
            "catalog": {
                "count": int(catalog.get("count", len(catalog.get("items") or []))),
                "hf_cache": catalog.get("hf_cache"),
            },
            "reasons": list(state["reasons"]),
        }

    def engine_status(self) -> dict[str, Any]:
        state = self._snapshot()
        descriptor = state["model"]
        return {
            "type": "model-driven-video-runtime",
            "selected_model": descriptor.to_public_dict() if descriptor is not None else None,
            "runtime": state["runtime"],
            "gpu_lease": self.services.gpu_lease(),
            "queue": self.services.queue_snapshot(),
            "output_dir": str(self.services.resolve_output_dir(state["config"])),
            "ready": state["engine_ready"],
            "reasons": list(state["reasons"]),
        }

    def unload(self) -> dict[str, Any]:
        self.register_backends()
        self.resolver.unload_all()
        return {"ok": True, "message": "DuckMotion runtime resources unloaded."}
