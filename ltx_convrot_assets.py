"""Asset/config discovery for LTX-2.5 INT8 ConvRot checkpoints.

ConvRot is an internal checkpoint format.  Public model payloads continue to expose
only model identity and capabilities; this module is used by discovery/readiness and
the isolated worker to locate the model's companion recipe and support weights.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


ASSET_KINDS = ("text_encoder", "latent_upscaler", "video_vae", "audio_vae")


def is_ltx25_convrot_path(value: str | Path) -> bool:
    path = Path(value).expanduser()
    name = path.name.lower()
    return (
        path.suffix.lower() == ".safetensors"
        and "convrot" in name
        and ("ltx-2.5" in name or "ltx25" in name or "ltx2.5" in name or "ltx" in name)
    )


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def find_companion_config(checkpoint: str | Path) -> Path | None:
    checkpoint_path = Path(checkpoint).expanduser()
    parent = checkpoint_path.parent
    if not parent.exists():
        return None
    try:
        candidates = sorted(parent.glob("*.json"), key=lambda p: p.name.lower())
    except OSError:
        return None

    checkpoint_name = checkpoint_path.name.lower()
    for candidate in candidates:
        payload = read_json(candidate)
        if not payload:
            continue
        values = (value.lower().replace("\\", "/") for value in _walk_strings(payload))
        if any(checkpoint_name == Path(value).name.lower() for value in values):
            return candidate

    # Civitai companion configs are often named for the recipe rather than the
    # checkpoint.  If there is exactly one JSON beside a ConvRot checkpoint it
    # is still a useful deterministic candidate, but readiness validates that it
    # actually describes all required assets before execution.
    if len(candidates) == 1:
        return candidates[0]
    return None


def extract_asset_names(config: dict[str, Any], checkpoint_name: str = "") -> dict[str, str]:
    strings = [value.replace("\\", "/") for value in _walk_strings(config)]
    safetensors = [Path(value).name for value in strings if value.lower().endswith(".safetensors")]
    checkpoint_lower = checkpoint_name.lower()

    def first(predicate) -> str:
        for name in safetensors:
            lower = name.lower()
            if checkpoint_lower and lower == checkpoint_lower:
                continue
            if predicate(lower):
                return name
        return ""

    return {
        "text_encoder": first(lambda v: "gemma" in v and "ltx" in v and "convrot" in v),
        "latent_upscaler": first(lambda v: "spatial-upscaler" in v or "latent-upscaler" in v),
        "video_vae": first(lambda v: "video-vae" in v),
        "audio_vae": first(lambda v: "audio-vae" in v),
    }


def candidate_search_roots(
    checkpoint: str | Path,
    *,
    models_dir: str | None = None,
) -> list[Path]:
    checkpoint_path = Path(checkpoint).expanduser()
    roots: list[Path] = [checkpoint_path.parent]
    for raw in (
        models_dir,
        os.getenv("DUCKMOTION_MODELS_DIR"),
        os.getenv("WEBBDUCK_MODELS_DIR"),
    ):
        if raw:
            path = Path(raw).expanduser()
            if path not in roots:
                roots.append(path)
    return roots


def resolve_named_asset(name: str, roots: Iterable[Path]) -> Path | None:
    if not name:
        return None
    target = Path(name).name.lower()
    for root in roots:
        root = Path(root).expanduser()
        direct = root / Path(name).name
        if direct.exists() and direct.is_file():
            return direct.resolve()
        if not root.exists() or not root.is_dir():
            continue
        try:
            for candidate in root.rglob(Path(name).name):
                if candidate.is_file() and candidate.name.lower() == target:
                    return candidate.resolve()
        except OSError:
            continue
    return None


def inspect_convrot_assets(
    checkpoint: str | Path,
    *,
    models_dir: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).expanduser()
    config_path = find_companion_config(checkpoint_path)
    config = read_json(config_path) if config_path else {}
    names = extract_asset_names(config, checkpoint_path.name) if config else {kind: "" for kind in ASSET_KINDS}
    roots = candidate_search_roots(checkpoint_path, models_dir=models_dir)
    resolved: dict[str, str | None] = {}
    missing: list[str] = []
    for kind in ASSET_KINDS:
        name = names.get(kind) or ""
        path = resolve_named_asset(name, roots) if name else None
        resolved[kind] = str(path) if path else None
        if not path:
            missing.append(name or f"<{kind} not declared in companion config>")

    checkpoint_present = checkpoint_path.exists() and checkpoint_path.is_file()
    if not checkpoint_present:
        missing.insert(0, checkpoint_path.name or str(checkpoint_path))
    if config_path is None:
        missing.insert(0, "<companion JSON config>")

    return {
        "ready": checkpoint_present and config_path is not None and not missing,
        "checkpoint": str(checkpoint_path),
        "config_path": str(config_path) if config_path else None,
        "asset_names": names,
        "assets": resolved,
        "missing": missing,
        "search_roots": [str(root) for root in roots],
    }
