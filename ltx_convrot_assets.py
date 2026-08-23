"""Recipe and asset discovery for LTX-2.5 INT8 ConvRot checkpoints.

Checkpoint identity, recipe identity, and product/display names are deliberately
separate. This module never selects behavior because a checkpoint is named
REDGraft (or any other brand). A checkpoint is detected as LTX-2.5 ConvRot by
format/metadata hints; a declarative companion recipe then describes the support
assets and must map to a DuckMotion-supported execution profile.

Companion JSON is evidence/configuration only. DuckMotion never executes an
arbitrary Comfy graph.
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
SUPPORTED_EXECUTION_PROFILE = "ltx25_convrot_two_stage_av"
PROFILE_REQUIRED_NODES = {
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "ConditioningZeroOut",
    "LTXVConditioning",
    "LTXVEmptyLatentAudio",
    "LTXVConcatAVLatent",
    "SamplerCustomAdvanced",
    "LTXVSeparateAVLatent",
    "LTXVCropGuides",
    "LTXVLatentUpsampler",
    "LatentUpscaleBy",
    "LatentUpscaleModelLoader",
    "VAEDecodeTiled",
    "LTXVAudioVAEDecode",
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
    """Best-effort format detection with structural metadata preferred.

    Filename tokens remain a compatibility fallback because many community
    safetensors omit useful format metadata, but no model/brand name selects a
    recipe or execution behavior.
    """
    path = Path(value).expanduser()
    if path.suffix.lower() != ".safetensors":
        return False

    name = path.name.lower()
    if path.exists() and path.is_file():
        header = _safetensors_header_text(path)
        if "convrot" in header and any(marker in header for marker in ("ltx", "ltxv")):
            return True

    ltx_name = any(marker in name for marker in ("ltx-2.5", "ltx25", "ltx2.5", "ltx"))
    return ltx_name and "convrot" in name


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_dicts(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_dicts(item)


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

    # Some community bundles rename the checkpoint after exporting the workflow.
    # A single sidecar JSON is a deterministic candidate, but recipe validation
    # below still has to prove it maps to a supported execution profile.
    if len(candidates) == 1:
        return candidates[0]
    return None


def _node_types(config: dict[str, Any]) -> set[str]:
    types: set[str] = set()
    for item in _walk_dicts(config):
        value = item.get("type")
        if isinstance(value, str):
            types.add(value)
    return types


def execution_profile(config: dict[str, Any]) -> str | None:
    """Map declarative workflow evidence to a supported execution profile."""
    node_types = _node_types(config)
    if PROFILE_REQUIRED_NODES.issubset(node_types):
        return SUPPORTED_EXECUTION_PROFILE
    return None


def _declared_models(config: dict[str, Any]) -> list[dict[str, str]]:
    """Extract declarative model records embedded by Comfy workflow exports."""
    declarations: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in _walk_dicts(config):
        models = item.get("models")
        if not isinstance(models, list):
            continue
        for model in models:
            if not isinstance(model, dict):
                continue
            name = str(model.get("name") or "").replace("\\", "/").strip()
            url = str(model.get("url") or "").strip()
            directory = str(model.get("directory") or "").replace("\\", "/").strip()
            if not name:
                continue
            key = (Path(name).name.lower(), url)
            if key in seen:
                continue
            seen.add(key)
            declarations.append({"name": Path(name).name, "url": url, "directory": directory})
    return declarations


def extract_asset_manifest(config: dict[str, Any], checkpoint_name: str = "") -> dict[str, dict[str, str]]:
    """Extract required asset names and optional source URLs from a recipe."""
    declarations = _declared_models(config)
    declared_by_name = {item["name"].lower(): item for item in declarations}
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

    names = {
        "text_encoder": first(lambda v: "gemma" in v and "ltx" in v),
        "latent_upscaler": first(lambda v: "spatial-upscaler" in v or "latent-upscaler" in v),
        "video_vae": first(lambda v: "video-vae" in v),
        "audio_vae": first(lambda v: "audio-vae" in v),
    }
    manifest: dict[str, dict[str, str]] = {}
    for kind, name in names.items():
        declaration = declared_by_name.get(name.lower(), {}) if name else {}
        manifest[kind] = {
            "name": name,
            "url": str(declaration.get("url") or ""),
            "directory": str(declaration.get("directory") or ""),
        }
    return manifest


def extract_asset_names(config: dict[str, Any], checkpoint_name: str = "") -> dict[str, str]:
    """Compatibility helper retained for callers/tests that only need names."""
    manifest = extract_asset_manifest(config, checkpoint_name)
    return {kind: value.get("name", "") for kind, value in manifest.items()}


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
    profile = execution_profile(config) if config else None
    manifest = extract_asset_manifest(config, checkpoint_path.name) if config else {
        kind: {"name": "", "url": "", "directory": ""} for kind in ASSET_KINDS
    }
    names = {kind: manifest[kind].get("name", "") for kind in ASSET_KINDS}
    roots = candidate_search_roots(checkpoint_path, models_dir=models_dir)

    resolved: dict[str, str | None] = {}
    missing: list[str] = []
    for kind in ASSET_KINDS:
        name = names.get(kind) or ""
        path = resolve_named_asset(name, roots) if name else None
        resolved[kind] = str(path) if path else None
        if not path:
            missing.append(name or f"<{kind} not declared in companion recipe>")

    checkpoint_present = checkpoint_path.exists() and checkpoint_path.is_file()
    if not checkpoint_present:
        missing.insert(0, checkpoint_path.name or str(checkpoint_path))
    if config_path is None:
        missing.insert(0, "<companion recipe JSON>")
    elif profile is None:
        missing.insert(0, "<unsupported LTX ConvRot execution recipe>")

    return {
        "ready": checkpoint_present and config_path is not None and profile is not None and not missing,
        "checkpoint": str(checkpoint_path),
        "config_path": str(config_path) if config_path else None,
        "execution_profile": profile,
        "asset_manifest": manifest,
        "asset_names": names,
        "assets": resolved,
        "missing": missing,
        "search_roots": [str(root) for root in roots],
        "asset_cache": str(convrot_asset_cache_root()),
    }
