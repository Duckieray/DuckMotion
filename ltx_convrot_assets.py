"""Asset/config discovery for LTX-2.5 INT8 ConvRot checkpoints.

ConvRot is an internal checkpoint format. Public model payloads continue to expose
only model identity and capabilities; this module is used by discovery/readiness and
the isolated worker to locate the model's support weights.

Known REDGraft checkpoints use a fixed DuckMotion recipe audited from the original
companion workflow. The workflow JSON is therefore optional at runtime: if present
it is still used as declarative evidence, but normal users do not have to keep a
Comfy workflow beside the checkpoint just to run the supported recipe.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


ASSET_KINDS = ("text_encoder", "latent_upscaler", "video_vae", "audio_vae")
MODEL_CATEGORY_DIRS = {
    "checkpoints",
    "checkpoint",
    "diffusion_models",
    "diffusion-models",
    "unet",
}

# The audited REDGraft workflow declares these exact support assets. They are
# published by Lightricks; setup downloads them into DuckMotion's private asset
# cache when they are not already available in the user's shared model roots.
REDGRAFT_ASSET_SOURCES: dict[str, dict[str, str]] = {
    "text_encoder": {
        "name": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "repo_id": "Lightricks/LTX-2.5",
        "filename": "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    },
    "latent_upscaler": {
        "name": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "repo_id": "Lightricks/LTX-2.3",
        "filename": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    },
    "video_vae": {
        "name": "ltx-2.5-video-vae-conv-bf16.safetensors",
        "repo_id": "Lightricks/LTX-2.5",
        "filename": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    },
    "audio_vae": {
        "name": "ltx-2.5-audio-vae-bf16.safetensors",
        "repo_id": "Lightricks/LTX-2.5",
        "filename": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    },
}


def asset_cache_root() -> Path:
    raw = str(os.getenv("DUCKMOTION_ASSET_CACHE") or "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".cache" / "duckmotion" / "assets"


def convrot_asset_cache_root() -> Path:
    return asset_cache_root() / "ltx25_convrot"


def _safetensors_header_text(path: Path) -> str:
    """Read only the small safetensors JSON header, never tensor payload bytes."""
    try:
        with path.open("rb") as handle:
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                return ""
            header_size = int.from_bytes(raw_size, "little", signed=False)
            if header_size <= 0 or header_size > 32 * 1024 * 1024:
                return ""
            raw_header = handle.read(header_size)
        payload = json.loads(raw_header.decode("utf-8"))
    except Exception:
        return ""
    try:
        return json.dumps(payload, sort_keys=True).lower()
    except Exception:
        return str(payload).lower()


def is_ltx25_convrot_path(value: str | Path) -> bool:
    path = Path(value).expanduser()
    if path.suffix.lower() != ".safetensors":
        return False

    name = path.name.lower()
    ltx_name = any(marker in name for marker in ("ltx-2.5", "ltx25", "ltx2.5", "ltx"))
    if ltx_name and ("convrot" in name or "redgraft" in name):
        return True

    if not path.exists() or not path.is_file():
        return False
    header = _safetensors_header_text(path)
    return (
        "convrot" in header
        and any(marker in f"{name}\n{header}" for marker in ("ltx", "ltxv", "redgraft"))
    )


def is_redgraft_checkpoint(value: str | Path) -> bool:
    path = Path(value).expanduser()
    return is_ltx25_convrot_path(path) and "redgraft" in path.name.lower()


def builtin_recipe_asset_names(checkpoint: str | Path) -> dict[str, str] | None:
    """Return the fixed audited asset contract for known REDGraft checkpoints."""
    if not is_redgraft_checkpoint(checkpoint):
        return None
    return {kind: spec["name"] for kind, spec in REDGRAFT_ASSET_SOURCES.items()}


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


def _inferred_model_root(checkpoint_path: Path) -> Path | None:
    for ancestor in checkpoint_path.parent.parents:
        if ancestor.name.lower() in MODEL_CATEGORY_DIRS:
            return ancestor.parent
    if checkpoint_path.parent.name.lower() in MODEL_CATEGORY_DIRS:
        return checkpoint_path.parent.parent
    return None


def candidate_search_roots(
    checkpoint: str | Path,
    *,
    models_dir: str | None = None,
) -> list[Path]:
    checkpoint_path = Path(checkpoint).expanduser()
    roots: list[Path] = [checkpoint_path.parent]
    inferred_root = _inferred_model_root(checkpoint_path)
    if inferred_root is not None and inferred_root not in roots:
        roots.append(inferred_root)
    for raw in (
        models_dir,
        os.getenv("DUCKMOTION_MODELS_DIR"),
        os.getenv("WEBBDUCK_MODELS_DIR"),
    ):
        if raw:
            path = Path(raw).expanduser()
            if path not in roots:
                roots.append(path)
    cache = convrot_asset_cache_root()
    if cache not in roots:
        roots.append(cache)
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
    builtin_names = builtin_recipe_asset_names(checkpoint_path)

    if config:
        names = extract_asset_names(config, checkpoint_path.name)
        recipe_source = "companion_json"
    elif builtin_names:
        names = builtin_names
        recipe_source = "builtin:redgraft_ltx25_fast2k"
    else:
        names = {kind: "" for kind in ASSET_KINDS}
        recipe_source = None

    roots = candidate_search_roots(checkpoint_path, models_dir=models_dir)
    resolved: dict[str, str | None] = {}
    missing: list[str] = []
    for kind in ASSET_KINDS:
        name = names.get(kind) or ""
        path = resolve_named_asset(name, roots) if name else None
        resolved[kind] = str(path) if path else None
        if not path:
            missing.append(name or f"<{kind} not declared by a supported recipe>")

    checkpoint_present = checkpoint_path.exists() and checkpoint_path.is_file()
    if not checkpoint_present:
        missing.insert(0, checkpoint_path.name or str(checkpoint_path))
    if recipe_source is None:
        missing.insert(0, "<supported companion recipe>")

    return {
        "ready": checkpoint_present and recipe_source is not None and not missing,
        "checkpoint": str(checkpoint_path),
        "config_path": str(config_path) if config_path else None,
        "recipe_source": recipe_source,
        "asset_names": names,
        "assets": resolved,
        "missing": missing,
        "search_roots": [str(root) for root in roots],
        "asset_cache": str(convrot_asset_cache_root()),
    }
