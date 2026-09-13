"""Shared LoRA discovery and selection helpers for DuckMotion.

DuckMotion intentionally reuses WebbDuck's LoRA library root. Video LoRAs are
namespaced by model family beneath that root; LTX adapters live in ``ltx/``.
The browser sends logical adapter names and weights only. Absolute filesystem
paths are resolved server-side immediately before a backend launches its worker.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from job_runtime import VideoJobCoordinator
from model_runtime import VideoModelDescriptor


LTX_LORA_NAMESPACE = "ltx"
SUPPORTED_SUFFIXES = {".safetensors"}
MIN_LORA_WEIGHT = -4.0
MAX_LORA_WEIGHT = 4.0
LTX_LORA_BACKENDS = {"ltx25_isolated", "ltx25_convrot"}


def _first_existing(candidates: Iterable[Path]) -> Path:
    values = list(candidates)
    for candidate in values:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return values[0]


def resolve_webbduck_lora_root() -> Path:
    """Resolve the exact LoRA root WebbDuck uses whenever possible."""
    explicit = str(os.getenv("WEBBDUCK_LORA_DIR") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()

    try:
        from models import registry as webbduck_registry

        root = getattr(webbduck_registry, "LORA_ROOT", None)
        if root:
            return Path(root).expanduser().resolve()
    except Exception:
        pass

    models_dir = str(os.getenv("WEBBDUCK_MODELS_DIR") or "").strip()
    if models_dir:
        base = Path(models_dir).expanduser().resolve()
    else:
        base = Path.cwd().resolve()
    return _first_existing((base / "lora", base / "loras")).resolve()


def ltx_lora_root() -> Path:
    return resolve_webbduck_lora_root() / LTX_LORA_NAMESPACE


def supports_loras(descriptor: VideoModelDescriptor | None) -> bool:
    """Return whether the installed runtime can apply LoRAs to this model."""
    if descriptor is None:
        return False
    return bool(
        descriptor.supported
        and descriptor.architecture == "ltx25"
        and descriptor.backend in LTX_LORA_BACKENDS
    )


def _load_registry_metadata(root: Path) -> dict[str, Any]:
    path = root / "loras.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def discover_ltx_loras() -> list[dict[str, Any]]:
    """Discover LTX LoRAs from ``<WebbDuck LoRA root>/ltx`` recursively."""
    root = resolve_webbduck_lora_root()
    namespace_root = root / LTX_LORA_NAMESPACE
    if not namespace_root.exists():
        return []

    registry_meta = _load_registry_metadata(root)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(namespace_root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        name = path.stem
        if name in seen:
            # Match WebbDuck's stem-based registry behavior and avoid ambiguous
            # browser selections if duplicate filenames exist below ltx/.
            continue
        seen.add(name)
        relative = path.relative_to(root).as_posix()
        metadata = registry_meta.get(name) if isinstance(registry_meta.get(name), dict) else {}
        rows.append(
            {
                "name": name,
                "file": relative,
                "weight": _safe_weight(metadata.get("weight"), 1.0),
                "trigger": metadata.get("trigger"),
                "description": str(metadata.get("description") or ""),
            }
        )
    return rows


def public_lora_catalog(*, supported: bool) -> dict[str, Any]:
    items = discover_ltx_loras() if supported else []
    return {
        "supported": bool(supported),
        "namespace": LTX_LORA_NAMESPACE,
        "count": len(items),
        "items": items,
    }


def _safe_weight(value: Any, fallback: float = 1.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if not math.isfinite(result):
        return float(fallback)
    return result


def normalize_lora_selection(raw: Any, *, supported: bool) -> list[dict[str, Any]]:
    """Validate a WebbDuck-style ``[{name, weight}]`` LoRA selection."""
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        raise ValueError("LoRAs must be a list")
    if raw and not supported:
        raise ValueError("The selected video model does not support LoRAs")

    catalog = {item["name"]: item for item in discover_ltx_loras()}
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            name = item.strip()
            weight_value = None
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("model") or "").strip()
            weight_value = item.get("weight", item.get("strength"))
        else:
            raise ValueError("Each LoRA must be a name or an object with name/weight")
        if not name:
            raise ValueError("Each LoRA selection requires a name")
        if name in seen:
            continue
        entry = catalog.get(name)
        if entry is None:
            raise ValueError(f"Unknown LTX LoRA: {name}")
        weight = _safe_weight(weight_value, entry.get("weight", 1.0))
        if not MIN_LORA_WEIGHT <= weight <= MAX_LORA_WEIGHT:
            raise ValueError(
                f"LoRA weight for '{name}' must be between {MIN_LORA_WEIGHT:g} and {MAX_LORA_WEIGHT:g}"
            )
        normalized.append(
            {
                "name": name,
                "file": str(entry["file"]),
                "weight": weight,
                "trigger": entry.get("trigger"),
            }
        )
        seen.add(name)
    return normalized


def materialize_lora_paths(raw: Any) -> list[dict[str, Any]]:
    """Resolve a normalized selection to trusted absolute paths for workers."""
    root = resolve_webbduck_lora_root().resolve()
    namespace = (root / LTX_LORA_NAMESPACE).resolve()
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        raise ValueError("LoRAs must be a list")

    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Normalized LoRA entries must be objects")
        name = str(item.get("name") or "").strip()
        relative = str(item.get("file") or "").strip()
        if not name or not relative:
            raise ValueError("Normalized LoRA entry is missing name/file")
        path = (root / relative).resolve()
        try:
            path.relative_to(namespace)
        except ValueError as exc:
            raise ValueError(f"LoRA '{name}' is outside the LTX LoRA namespace") from exc
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"LTX LoRA file is missing: {name}")
        result.append(
            {
                "name": name,
                "path": str(path),
                "weight": _safe_weight(item.get("weight"), 1.0),
                "trigger": item.get("trigger"),
            }
        )
    return result


class LoraAwareVideoJobCoordinator(VideoJobCoordinator):
    """Video coordinator that persists validated adapter selection with a job."""

    def _normalize_params(
        self,
        descriptor: VideoModelDescriptor,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        params = super()._normalize_params(descriptor, request)
        loras = normalize_lora_selection(
            request.get("loras"),
            supported=supports_loras(descriptor),
        )
        if loras:
            params["loras"] = loras
        return params
