"""Model-driven video runtime contracts for DuckMotion.

DuckMotion's UI should expose models and capabilities, not engine families.
Architecture, checkpoint format and backend identifiers in this module are
internal routing data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping

from ltx_convrot_assets import is_ltx25_convrot_path


UNKNOWN_ARCHITECTURE = "unknown"
UNSUPPORTED_BACKEND = "unsupported"


@dataclass(frozen=True)
class VideoCapabilities:
    text_to_video: bool = False
    image_to_video: bool = False
    video_to_video: bool = False
    audio_output: bool = False
    negative_prompt: bool = False
    source_image_required: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class VideoModelDescriptor:
    name: str
    source: str
    architecture: str = UNKNOWN_ARCHITECTURE
    backend: str = UNSUPPORTED_BACKEND
    capabilities: VideoCapabilities = field(default_factory=VideoCapabilities)
    defaults: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    detection: dict[str, Any] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        return backend_is_implemented(self.backend) and (
            self.capabilities.text_to_video or self.capabilities.image_to_video
        )

    def to_internal_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["supported"] = self.supported
        return value

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "capabilities": self.capabilities.to_dict(),
            "defaults": dict(self.defaults),
            "constraints": dict(self.constraints),
            "supported": self.supported,
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _tokens(config: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("_class_name", "architectures", "model_type"):
        value = config.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    return " ".join(parts).lower()


def _gguf_pair_role(path: Path) -> str | None:
    stem = path.stem
    match = re.search(r"(?i)(?:[_ .-]?)([hl])$", stem)
    return match.group(1).upper() if match else None


def _source_tokens(source: str, name: str | None = None) -> tuple[str, dict[str, Any]]:
    path = Path(source).expanduser()
    display_name = str(name or "").lower()
    if path.exists() and path.is_file():
        suffix = path.suffix.lower()
        filename_tokens = " ".join(
            part for part in (str(path).lower(), display_name, path.name.lower()) if part
        )
        detection: dict[str, Any] = {
            "method": "local_single_file",
            "confidence": "high",
            "format": suffix.lstrip(".") or "single_file",
        }
        if suffix == ".gguf":
            detection["format"] = "gguf"
            role = _gguf_pair_role(path)
            if role:
                detection["pair_role"] = role
        elif suffix == ".safetensors" and is_ltx25_convrot_path(path):
            detection["format"] = "int8_convrot"
        return filename_tokens, detection

    if path.exists() and path.is_dir():
        index = _read_json(path / "model_index.json")
        transformer = _read_json(path / "transformer" / "config.json")
        tokens = " ".join(
            part
            for part in (
                _tokens(index),
                _tokens(transformer),
                display_name,
                path.name.lower(),
            )
            if part
        )
        return tokens, {
            "method": "local_config",
            "confidence": "high" if index or transformer else "medium",
            "format": "diffusers",
            "dual_transformer": bool(
                (path / "transformer_2").exists() or index.get("transformer_2")
            ),
        }
    tokens = " ".join(part for part in (str(source).lower(), display_name) if part)
    detection = {
        "method": "source_name",
        "confidence": "medium",
        "format": "diffusers",
    }
    if ".safetensors" in tokens and "ltx" in tokens and (
        "convrot" in tokens or "redgraft" in tokens
    ):
        detection["format"] = "int8_convrot"
    return tokens, detection


def detect_video_architecture(
    source: str,
    *,
    name: str | None = None,
) -> tuple[str, VideoCapabilities, dict[str, Any]]:
    """Infer model family and workflows runnable by the installed backend."""
    tokens, detection = _source_tokens(str(source or ""), name=name)

    if "ltx-2.5" in tokens or "ltx2.5" in tokens or "ltx25" in tokens or "ltx2" in tokens:
        variant = "convrot" if detection.get("format") == "int8_convrot" else "distilled"
        return (
            "ltx25",
            VideoCapabilities(
                text_to_video=True,
                image_to_video=True,
                video_to_video=False,
                audio_output=True,
                negative_prompt=False,
                source_image_required=False,
            ),
            {**detection, "variant": variant},
        )

    if "wan" in tokens:
        is_ti2v = (
            "ti2v" in tokens
            or "text-image-to-video" in tokens
            or "text_image_to_video" in tokens
        )
        is_i2v = (
            "imagetovideo" in tokens
            or "image-to-video" in tokens
            or "image_to_video" in tokens
            or "i2v" in tokens
        )
        is_t2v = (
            is_ti2v
            or "texttovideo" in tokens
            or "text-to-video" in tokens
            or "text_to_video" in tokens
            or "t2v" in tokens
            or "wanpipeline" in tokens
        )
        if is_i2v and is_t2v and not is_ti2v:
            capabilities = VideoCapabilities(
                text_to_video=True,
                image_to_video=True,
                negative_prompt=True,
                source_image_required=False,
            )
            variant = "t2v_i2v"
        elif is_ti2v:
            capabilities = VideoCapabilities(
                text_to_video=True,
                image_to_video=False,
                negative_prompt=True,
                source_image_required=False,
            )
            variant = "ti2v"
            detection = {
                **detection,
                "upstream_image_to_video": True,
                "runtime_image_to_video": False,
            }
        elif is_t2v and not is_i2v:
            capabilities = VideoCapabilities(
                text_to_video=True,
                negative_prompt=True,
                source_image_required=False,
            )
            variant = "t2v"
        else:
            capabilities = VideoCapabilities(
                image_to_video=True,
                negative_prompt=True,
                source_image_required=True,
            )
            variant = "i2v"
        return "wan22", capabilities, {**detection, "variant": variant}

    return UNKNOWN_ARCHITECTURE, VideoCapabilities(), {
        **detection,
        "confidence": "none",
    }


def backend_for_architecture(architecture: str | None) -> str:
    return {
        "wan22": "wan_diffusers",
        "ltx25": "ltx25_isolated",
    }.get((architecture or "").lower(), UNSUPPORTED_BACKEND)


def backend_for_model(
    architecture: str | None,
    capabilities: VideoCapabilities,
    detection: Mapping[str, Any] | None = None,
) -> str:
    architecture = (architecture or "").lower()
    if architecture == "wan22" and not (
        capabilities.text_to_video or capabilities.image_to_video
    ):
        return UNSUPPORTED_BACKEND
    if architecture == "ltx25" and str((detection or {}).get("format") or "").lower() == "int8_convrot":
        return "ltx25_convrot"
    return backend_for_architecture(architecture)


_IMPLEMENTED_BACKENDS = {"wan_diffusers", "ltx25_isolated", "ltx25_convrot"}


def backend_is_implemented(backend: str | None) -> bool:
    return (backend or "") in _IMPLEMENTED_BACKENDS


def constraints_for_architecture(
    architecture: str | None,
    detection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    architecture = (architecture or "").lower()
    if architecture == "ltx25":
        constraints = {
            "dimension_multiple": 64,
            "frame_count_modulo": 8,
            "frame_count_remainder": 1,
            "generation_stages": 2,
            "sampling_schedule_locked": True,
        }
        if str((detection or {}).get("format") or "").lower() == "int8_convrot":
            constraints["checkpoint_recipe_required"] = True
        return constraints
    if architecture == "wan22":
        return {
            "dimension_multiple": 16,
            "frame_count_modulo": 4,
            "frame_count_remainder": 1,
        }
    return {}


def defaults_for_architecture(architecture: str | None) -> dict[str, Any]:
    architecture = (architecture or "").lower()
    if architecture == "ltx25":
        return {
            "width": 1536,
            "height": 1024,
            "num_frames": 121,
            "fps": 24,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
        }
    if architecture == "wan22":
        return {
            "width": 832,
            "height": 480,
            "num_frames": 81,
            "fps": 16,
            "num_inference_steps": 30,
            "guidance_scale": 5.0,
        }
    return {}


def defaults_for_model(
    architecture: str | None,
    name: str,
    detection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = defaults_for_architecture(architecture)
    architecture = (architecture or "").lower()
    text = str(name or "").lower()
    variant = str((detection or {}).get("variant") or "").lower()

    if architecture == "ltx25" and variant == "convrot":
        defaults.update(
            {
                "width": 1152,
                "height": 768,
                "num_frames": 241,
                "fps": 24,
                "num_inference_steps": 8,
                "guidance_scale": 1.0,
            }
        )

    if architecture == "wan22" and variant == "ti2v":
        defaults.update(
            {
                "width": 1280,
                "height": 704,
                "num_frames": 121,
                "fps": 24,
                "num_inference_steps": 50,
                "guidance_scale": 5.0,
            }
        )

    if architecture == "wan22" and "turbo" in text:
        defaults.update(
            {
                "num_frames": 121,
                "fps": 24,
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
            }
        )
    return defaults


def describe_video_model(
    source: str,
    *,
    name: str | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> VideoModelDescriptor:
    display_name = name or Path(source).name or source
    architecture, capabilities, detection = detect_video_architecture(
        source,
        name=display_name,
    )
    effective_defaults = defaults_for_model(architecture, display_name, detection)
    effective_defaults.update(dict(defaults or {}))
    return VideoModelDescriptor(
        name=display_name,
        source=source,
        architecture=architecture,
        backend=backend_for_model(architecture, capabilities, detection),
        capabilities=capabilities,
        defaults=effective_defaults,
        constraints=constraints_for_architecture(architecture, detection),
        detection=detection,
    )


class VideoBackend(ABC):
    backend_id: str

    @abstractmethod
    def can_handle(self, descriptor: VideoModelDescriptor) -> bool:
        """Return whether this runtime adapter can execute the descriptor."""

    @abstractmethod
    def generate(
        self,
        descriptor: VideoModelDescriptor,
        request: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Execute a video generation request."""

    def readiness(self, descriptor: VideoModelDescriptor) -> dict[str, Any]:
        ready = bool(self.can_handle(descriptor))
        return {
            "ready": ready,
            "reason": None if ready else f"Installed runtime cannot handle video model '{descriptor.name}'.",
        }

    def unload(self) -> None:
        """Release runtime resources if loaded."""


class VideoBackendResolver:
    def __init__(self) -> None:
        self._backends: dict[str, VideoBackend] = {}

    def register(self, backend: VideoBackend) -> None:
        backend_id = str(getattr(backend, "backend_id", "") or "").strip()
        if not backend_id:
            raise ValueError("Video backend must define backend_id")
        self._backends[backend_id] = backend

    def unregister(self, backend_id: str) -> None:
        self._backends.pop(str(backend_id), None)

    def resolve(self, descriptor: VideoModelDescriptor) -> VideoBackend:
        preferred = self._backends.get(descriptor.backend)
        if preferred is not None and preferred.can_handle(descriptor):
            return preferred
        for backend in self._backends.values():
            if backend.can_handle(descriptor):
                return backend
        raise LookupError(
            f"No DuckMotion backend is registered for model '{descriptor.name}' "
            f"(backend={descriptor.backend!r})."
        )

    def readiness(self, descriptor: VideoModelDescriptor) -> dict[str, Any]:
        backend = self.resolve(descriptor)
        payload = backend.readiness(descriptor)
        if not isinstance(payload, dict):
            return {"ready": False, "reason": "Backend returned an invalid readiness payload."}
        return {
            **payload,
            "ready": bool(payload.get("ready")),
            "reason": payload.get("reason"),
        }

    def ids(self) -> tuple[str, ...]:
        return tuple(self._backends.keys())

    def unload_all(self) -> None:
        errors: list[Exception] = []
        for backend in tuple(self._backends.values()):
            try:
                backend.unload()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"Failed to unload {len(errors)} DuckMotion backend(s): {errors[0]}")


backend_resolver = VideoBackendResolver()
