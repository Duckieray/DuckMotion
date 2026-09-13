"""Recipe and asset discovery for LTX-2.5 INT8 ConvRot checkpoints.

Checkpoint identity, execution recipe, and product/display names are deliberately
separate. This module detects the checkpoint format, resolves a compatible
execution profile through DuckMotion's generic recipe registry, and resolves the
assets declared by the companion recipe.

A companion may use DuckMotion's small declarative ``duckmotion_recipe`` schema
or be an exported workflow that an adapter can map to a supported profile.
DuckMotion never executes arbitrary companion/workflow code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from model_provenance import cached_provenance, cached_recipe_paths
from model_recipes import (
    LTX25_CONVROT_TWO_STAGE_AV,
    execution_profiles,
    profile_from_explicit_manifest,
)


ARCHITECTURE = "ltx25"
SOURCE_FORMAT = "int8_convrot"
SUPPORTED_EXECUTION_PROFILE = LTX25_CONVROT_TWO_STAGE_AV.profile_id
PROFILE_REQUIRED_NODES = set(LTX25_CONVROT_TWO_STAGE_AV.evidence_node_types)
ASSET_KINDS = tuple(LTX25_CONVROT_TWO_STAGE_AV.required_assets)
MODEL_CATEGORY_DIRS = {
    "checkpoints",
    "checkpoint",
    "diffusion_models",
    "diffusion-models",
    "unet",
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


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
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


def _local_companion_candidates(checkpoint_path: Path) -> list[Path]:
    parent = checkpoint_path.parent
    if not parent.exists():
        return []
    try:
        candidates = sorted(parent.glob("*.json"), key=lambda p: p.name.lower())
    except OSError:
        return []

    checkpoint_name = checkpoint_path.name.lower()
    exact: list[Path] = []
    for candidate in candidates:
        payload = read_json(candidate)
        if not payload:
            continue
        values = (value.lower().replace("\\", "/") for value in _walk_strings(payload))
        if any(checkpoint_name == Path(value).name.lower() for value in values):
            exact.append(candidate)
    if exact:
        return exact + [candidate for candidate in candidates if candidate not in exact]

    # Community bundles sometimes rename a checkpoint after exporting its recipe.
    # A single sidecar is deterministic evidence, but profile validation below
    # still has to prove that the recipe is supported.
    if len(candidates) == 1:
        return candidates
    return []


def companion_config_candidates(checkpoint: str | Path) -> tuple[Path, ...]:
    checkpoint_path = Path(checkpoint).expanduser()
    candidates = _local_companion_candidates(checkpoint_path)
    for candidate in cached_recipe_paths(checkpoint_path):
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def find_companion_config(checkpoint: str | Path) -> Path | None:
    candidates = companion_config_candidates(checkpoint)
    supported: list[Path] = []
    for candidate in candidates:
        payload = read_json(candidate)
        if payload and execution_profile(payload):
            supported.append(candidate)
    if len(supported) == 1:
        return supported[0]
    if len(supported) > 1:
        return None
    return candidates[0] if len(candidates) == 1 else None


def _node_types(config: Mapping[str, Any]) -> set[str]:
    types: set[str] = set()
    for item in _walk_dicts(config):
        value = item.get("type")
        if isinstance(value, str):
            types.add(value)
    return types


def _workflow_literals(config: Mapping[str, Any]) -> set[str]:
    return {value.strip().lower() for value in _walk_strings(config) if value.strip()}


def execution_profile(config: Mapping[str, Any]) -> str | None:
    """Resolve a supported profile from explicit manifest or proven workflow evidence."""
    explicit = profile_from_explicit_manifest(
        config,
        architecture=ARCHITECTURE,
        source_format=SOURCE_FORMAT,
    )
    if explicit is not None:
        return explicit.profile_id

    inferred = execution_profiles.infer_from_node_types(
        architecture=ARCHITECTURE,
        source_format=SOURCE_FORMAT,
        node_types=_node_types(config),
    )
    if inferred is None:
        return None
    required_literals = {value.strip().lower() for value in inferred.evidence_literals if value.strip()}
    if required_literals and not required_literals.issubset(_workflow_literals(config)):
        return None
    return inferred.profile_id


def is_ltx25_convrot_path(value: str | Path) -> bool:
    """Best-effort format detection; profile selection happens separately.

    Structural safetensors metadata and a compatible companion recipe are the
    strongest signals. Filename hints are compatibility fallbacks for community
    checkpoints that omit useful metadata. A brand token may be recognized as a
    weak *format* hint for known legacy files, but it never chooses a profile,
    asset set, or runtime behavior.
    """
    path = Path(value).expanduser()
    if path.suffix.lower() != ".safetensors":
        return False

    if path.exists() and path.is_file():
        header = _safetensors_header_text(path)
        if "convrot" in header and any(marker in header for marker in ("ltx", "ltxv")):
            return True

        companion = find_companion_config(path)
        if companion is not None:
            config = read_json(companion)
            if execution_profile(config):
                return True

    name = path.name.lower()
    ltx_name = any(marker in name for marker in ("ltx-2.5", "ltx25", "ltx2.5", "ltx"))
    if ltx_name and "convrot" in name:
        return True

    # Compatibility only for existing community artifacts whose filenames omit
    # the format marker. This is deliberately not used anywhere in recipe or
    # asset resolution.
    return ltx_name and "redgraft" in name


def _declared_models(config: Mapping[str, Any]) -> list[dict[str, str]]:
    """Extract declarative model records embedded by exported workflows."""
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


def _explicit_asset_manifest(config: Mapping[str, Any], profile_id: str) -> dict[str, dict[str, str]] | None:
    recipe = config.get("duckmotion_recipe")
    if not isinstance(recipe, Mapping):
        return None
    if str(recipe.get("profile") or "").strip() != profile_id:
        return None
    raw_assets = recipe.get("assets")
    if not isinstance(raw_assets, Mapping):
        return None

    manifest: dict[str, dict[str, str]] = {}
    for kind in ASSET_KINDS:
        raw = raw_assets.get(kind)
        if isinstance(raw, str):
            manifest[kind] = {"name": Path(raw).name, "url": "", "directory": ""}
        elif isinstance(raw, Mapping):
            name = str(raw.get("name") or "").replace("\\", "/").strip()
            manifest[kind] = {
                "name": Path(name).name if name else "",
                "url": str(raw.get("url") or "").strip(),
                "directory": str(raw.get("directory") or "").replace("\\", "/").strip(),
            }
        else:
            manifest[kind] = {"name": "", "url": "", "directory": ""}
    return manifest


def _workflow_asset_manifest(config: Mapping[str, Any], checkpoint_name: str = "") -> dict[str, dict[str, str]]:
    """Adapt an exported LTX workflow into the supported profile's asset roles."""
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


def _merge_profile_asset_defaults(
    profile_id: str,
    manifest: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    """Fill only missing standard declarations owned by the execution profile.

    A profile default may fill an empty role, or the source/directory for the same
    standard filename. A differently named custom recipe asset is preserved and
    never silently replaced with the profile default.
    """
    profile = execution_profiles.get(profile_id)
    result: dict[str, dict[str, str]] = {
        kind: {
            "name": str((manifest.get(kind) or {}).get("name") or "").strip(),
            "url": str((manifest.get(kind) or {}).get("url") or "").strip(),
            "directory": str((manifest.get(kind) or {}).get("directory") or "").strip(),
        }
        for kind in ASSET_KINDS
    }
    if profile is None:
        return result

    for kind in profile.required_assets:
        current = result.setdefault(kind, {"name": "", "url": "", "directory": ""})
        fallback = dict(profile.asset_defaults.get(kind) or {})
        fallback_name = str(fallback.get("name") or "").strip()
        if not current["name"] and fallback_name:
            current.update(
                {
                    "name": fallback_name,
                    "url": str(fallback.get("url") or "").strip(),
                    "directory": str(fallback.get("directory") or "").strip(),
                }
            )
            continue
        if current["name"] and fallback_name and current["name"].lower() == fallback_name.lower():
            if not current["url"]:
                current["url"] = str(fallback.get("url") or "").strip()
            if not current["directory"]:
                current["directory"] = str(fallback.get("directory") or "").strip()
    return result


def extract_asset_manifest(config: Mapping[str, Any], checkpoint_name: str = "") -> dict[str, dict[str, str]]:
    """Extract and normalize required asset roles for a supported companion."""
    profile_id = execution_profile(config)
    if not profile_id:
        return {kind: {"name": "", "url": "", "directory": ""} for kind in ASSET_KINDS}
    explicit = _explicit_asset_manifest(config, profile_id)
    raw_manifest = explicit if explicit is not None else _workflow_asset_manifest(config, checkpoint_name)
    return _merge_profile_asset_defaults(profile_id, raw_manifest)


def extract_asset_names(config: Mapping[str, Any], checkpoint_name: str = "") -> dict[str, str]:
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


def _select_recipe(checkpoint_path: Path) -> tuple[Path | None, dict[str, Any], str | None, bool]:
    candidates = companion_config_candidates(checkpoint_path)
    supported: list[tuple[Path, dict[str, Any], str]] = []
    first_payload: tuple[Path, dict[str, Any]] | None = None
    for candidate in candidates:
        payload = read_json(candidate)
        if not payload:
            continue
        if first_payload is None:
            first_payload = (candidate, payload)
        profile_id = execution_profile(payload)
        if profile_id:
            supported.append((candidate, payload, profile_id))
    if len(supported) == 1:
        path, payload, profile_id = supported[0]
        return path, payload, profile_id, False
    if len(supported) > 1:
        return None, {}, None, True
    if first_payload is not None:
        return first_payload[0], first_payload[1], None, False
    return None, {}, None, False


def inspect_convrot_assets(
    checkpoint: str | Path,
    *,
    models_dir: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).expanduser()
    config_path, config, profile_id, ambiguous_recipe = _select_recipe(checkpoint_path)
    profile = execution_profiles.get(profile_id)
    manifest = extract_asset_manifest(config, checkpoint_path.name) if profile else {
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
    if ambiguous_recipe:
        missing.insert(0, "<ambiguous supported companion recipes>")
    elif config_path is None:
        missing.insert(0, "<companion recipe JSON>")
    elif profile is None:
        missing.insert(0, "<unsupported execution recipe>")

    recipe = config.get("duckmotion_recipe") if isinstance(config, Mapping) else None
    provenance = cached_provenance(checkpoint_path)
    recipe_adapter = "duckmotion_manifest" if isinstance(recipe, Mapping) else ("workflow_adapter" if profile else None)
    recipe_origin = None
    if config_path is not None:
        recipe_origin = "local" if config_path.parent == checkpoint_path.parent else "provenance_cache"
    return {
        "ready": checkpoint_present and config_path is not None and profile is not None and not missing,
        "checkpoint": str(checkpoint_path),
        "config_path": str(config_path) if config_path else None,
        "execution_profile": profile.profile_id if profile else None,
        "recipe_adapter": recipe_adapter,
        "recipe_origin": recipe_origin,
        "provenance": provenance,
        "asset_manifest": manifest,
        "asset_names": names,
        "assets": resolved,
        "missing": missing,
        "search_roots": [str(root) for root in roots],
        "asset_cache": str(convrot_asset_cache_root()),
    }
