"""Generic local/Hugging Face model discovery for DuckMotion.

This module discovers model snapshots and describes their capabilities without
loading any video runtime. Wan and LTX can therefore coexist in one model list
while execution remains delegated to separate backend adapters.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from model_runtime import UNKNOWN_ARCHITECTURE, describe_video_model


VIDEO_ARCHITECTURES = {"wan22", "ltx25"}


def resolve_hf_cache_root() -> Path:
    explicit = os.getenv("WEBBDUCK_HF_CACHE_DIR")
    if explicit:
        value = Path(explicit).expanduser()
        return value if value.name.lower() == "hub" else value / "hub"

    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        raw = os.getenv(key)
        if raw:
            value = Path(raw).expanduser()
            return value if value.name.lower() == "hub" else value / "hub"

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def default_local_roots(config: Mapping[str, Any] | None = None) -> list[Path]:
    """Return architecture-neutral local model roots in preference order."""
    config = config or {}
    roots: list[Path] = []

    configured = str(config.get("models_dir") or "").strip()
    if configured:
        roots.append(Path(configured).expanduser())

    env_models = str(os.getenv("WEBBDUCK_MODELS_DIR") or "").strip()
    if env_models:
        base = Path(env_models).expanduser()
    else:
        base = Path.cwd()

    for candidate in (
        base / "checkpoint",
        base / "checkpoints",
        base,
    ):
        if candidate not in roots:
            roots.append(candidate)
    return roots


def _repo_display_name(repo: Path) -> str:
    return repo.name.removeprefix("models--").replace("--", "/")


def _snapshot_dirs(repo: Path) -> list[Path]:
    snapshots = repo / "snapshots"
    try:
        items = [item for item in snapshots.iterdir() if item.is_dir()]
    except OSError:
        return []

    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    return sorted(items, key=mtime, reverse=True)


def _public_item(*, descriptor, location: str) -> dict[str, Any]:
    payload = descriptor.to_public_dict()
    payload["location"] = location
    return payload


def _gguf_pair_identity(path: Path) -> tuple[str, str | None]:
    """Return a stable pair-family key and H/L role for historical Wan GGUF names."""
    stem = path.stem
    match = re.search(r"(?i)^(.*?)([_ .-]?)([hl])$", stem)
    if not match:
        return stem.lower(), None
    family = match.group(1).rstrip("_ .-") or stem
    return family.lower(), match.group(3).upper()


def _discover_gguf_sources(root: Path) -> list[Path]:
    """Return one selectable source per GGUF pair, preferring H over L."""
    try:
        files = sorted(root.rglob("*.gguf"), key=lambda p: str(p).lower())
    except OSError:
        return []

    grouped: dict[tuple[str, str], dict[str, Path]] = {}
    singles: list[Path] = []
    for path in files:
        family, role = _gguf_pair_identity(path)
        if role is None:
            singles.append(path)
            continue
        key = (str(path.parent.resolve()).lower(), family)
        grouped.setdefault(key, {})[role] = path

    selected = list(singles)
    for paths in grouped.values():
        selected.append(paths.get("H") or paths.get("L"))
    return sorted((path for path in selected if path is not None), key=lambda p: str(p).lower())


def _gguf_display_name(path: Path) -> str:
    family, _role = _gguf_pair_identity(path)
    # Keep the actual checkpoint identity visible while removing only the H/L
    # implementation suffix from paired transformer files.
    original = path.stem
    match = re.search(r"(?i)^(.*?)([_ .-]?)([hl])$", original)
    return (match.group(1).rstrip("_ .-") if match else original) or family


def discover_local_video_models(roots: Iterable[Path]) -> list[dict[str, Any]]:
    """Discover recognized video Diffusers directories and Wan GGUF files."""
    found: dict[str, dict[str, Any]] = {}
    seen_paths: set[str] = set()

    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.exists() or not root.is_dir():
            continue
        try:
            indexes = sorted(root.rglob("model_index.json"), key=lambda p: str(p).lower())
        except OSError:
            indexes = []

        for index in indexes:
            model_dir = index.parent
            resolved = str(model_dir.resolve())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            descriptor = describe_video_model(str(model_dir), name=model_dir.name)
            if descriptor.architecture not in VIDEO_ARCHITECTURES:
                continue
            key = f"local:{resolved}"
            found[key] = _public_item(descriptor=descriptor, location="local")

        for gguf_path in _discover_gguf_sources(root):
            resolved = str(gguf_path.resolve())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            descriptor = describe_video_model(
                resolved,
                name=_gguf_display_name(gguf_path),
            )
            if descriptor.architecture != "wan22":
                continue
            key = f"local:{resolved}"
            found[key] = _public_item(descriptor=descriptor, location="local")

    return sorted(found.values(), key=lambda item: str(item.get("name") or "").lower())


def discover_hf_video_models(hf_cache: Path) -> list[dict[str, Any]]:
    """Discover recognized video models in the standard Hugging Face hub cache."""
    cache = Path(hf_cache).expanduser()
    if not cache.exists() or not cache.is_dir():
        return []

    items: list[dict[str, Any]] = []
    try:
        repos = sorted(cache.glob("models--*"), key=lambda p: p.name.lower())
    except OSError:
        return []

    for repo in repos:
        display_name = _repo_display_name(repo)
        for snapshot in _snapshot_dirs(repo):
            if not (snapshot / "model_index.json").exists():
                continue
            descriptor = describe_video_model(str(snapshot), name=display_name)
            if descriptor.architecture not in VIDEO_ARCHITECTURES:
                continue
            item = _public_item(descriptor=descriptor, location="hf_cache")
            item["source"] = str(snapshot)
            item["repo_id"] = display_name
            items.append(item)
            break

    return sorted(items, key=lambda item: str(item.get("name") or "").lower())


def discover_configured_video_model(config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Describe an explicitly configured repo ID/path even if it is not cached yet."""
    config = config or {}
    source = str(config.get("model_id_or_path") or "").strip()
    if not source:
        return None
    descriptor = describe_video_model(source)
    if descriptor.architecture == UNKNOWN_ARCHITECTURE:
        return None
    return _public_item(descriptor=descriptor, location="configured")


def discover_video_models(
    config: Mapping[str, Any] | None = None,
    *,
    roots: Iterable[Path] | None = None,
    hf_cache: Path | None = None,
) -> dict[str, Any]:
    """Return a unified model list without exposing architecture/backend choices."""
    config = config or {}
    local_items = discover_local_video_models(
        roots if roots is not None else default_local_roots(config)
    )
    hf_items = discover_hf_video_models(hf_cache or resolve_hf_cache_root())
    configured = discover_configured_video_model(config)

    items: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for item in ([configured] if configured else []) + local_items + hf_items:
        if not item:
            continue
        source = str(item.get("source") or "")
        if source in seen_sources:
            continue
        seen_sources.add(source)
        items.append(item)

    return {
        "items": items,
        "count": len(items),
        "hf_cache": str(hf_cache or resolve_hf_cache_root()),
    }
