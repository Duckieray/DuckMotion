"""Model-driven video runtime contracts for DuckMotion.

DuckMotion's UI should expose models and capabilities, not engine families.
Architecture and backend identifiers in this module are internal routing data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


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
        """Whether this model is runnable in the current DuckMotion build."""
        return backend_is_implemented(self.backend) and (
            self.capabilities.text_to_video or self.capabilities.image_to_video
        )

    def to_internal_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["supported"] = self.supported
        return value

    def to_public_dict(self) -> dict[str, Any]:
        """Return the model-driven contract the UI actually needs."""
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


def _source_tokens(source: str) -> tuple[str, dict[str, Any]]:
    path = Path(source).expanduser()
    if path.exists() and path.is_dir():
        index = _read_json(path / "model_index.json")
        transformer = _read_json(path / "transformer" / "config.json")
        tokens = " ".join(
            part for part in (_tokens(index), _tokens(transformer), path.name.lower()) if part
        )
        return tokens, {
            "method": "local_config",
            "confidence": "high" if index or transformer else "medium",
        }
    return source.lower(), {"method": "source_name", "confidence": "medium"}


def detect_video_architecture(source: str) -> tuple[str, VideoCapabilities, dict[str, Any]]:
    """Infer model family and workflow capabilities from a selected model source."""
    tokens, detection = _source_tokens(str(source or ""))

    if "ltx-2.5" in tokens or "ltx2.5" in tokens or "ltx25" in tokens or "ltx2" in tokens:
        return (
            "ltx25",
            VideoCapabilities(
                text_to_video=True,
                image_to_video=True,
                video_to_video=True,
                audio_output=True,
                negative_prompt=True,
                source_image_required=False,
            ),
            detection,
        )

    if "wan" in tokens:
        is_i2v = "imagetovideo" in tokens or "image-to-video" in tokens or "i2v" in tokens
        is_t2v = "texttovideo" in tokens or "text-to-video" in tokens or "t2v" in tokens
        if is_t2v and not is_i2v:
            capabilities = VideoCapabilities(
                text_to_video=True,
                negative_prompt=True,
                source_image_required=False,
            )
        else:
            capabilities = VideoCapabilities(
                image_to_video=True,
                negative_prompt=True,
                source_image_required=True,
            )
        return "wan22", capabilities, detection

    return UNKNOWN_ARCHITECTURE, VideoCapabilities(), {
        **detection,
        "confidence": "none",
    }


def backend_for_architecture(architecture: str | None) -> str:
    return {
        "wan22": "wan_diffusers",
        # LTX starts isolated so newer runtime requirements cannot destabilize
        # the working Wan/WebbDuck environment.
        "ltx25": "ltx25_isolated",
    }.get((architecture or "").lower(), UNSUPPORTED_BACKEND)


# Discovery may know how a future model should run before that adapter exists.
# Only backends actually connected to live execution belong here.
_IMPLEMENTED_BACKENDS = {"wan_diffusers"}


def backend_is_implemented(backend: str | None) -> bool:
    return (backend or "") in _IMPLEMENTED_BACKENDS


def constraints_for_architecture(architecture: str | None) -> dict[str, Any]:
    architecture = (architecture or "").lower()
    if architecture == "ltx25":
        return {
            "dimension_multiple": 32,
            "frame_count_modulo": 8,
            "frame_count_remainder": 1,
        }
    if architecture == "wan22":
        return {"dimension_multiple": 16}
    return {}


def describe_video_model(
    source: str,
    *,
    name: str | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> VideoModelDescriptor:
    architecture, capabilities, detection = detect_video_architecture(source)
    display_name = name or Path(source).name or source
    return VideoModelDescriptor(
        name=display_name,
        source=source,
        architecture=architecture,
        backend=backend_for_architecture(architecture),
        capabilities=capabilities,
        defaults=dict(defaults or {}),
        constraints=constraints_for_architecture(architecture),
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

    def ids(self) -> tuple[str, ...]:
        return tuple(self._backends.keys())


backend_resolver = VideoBackendResolver()
