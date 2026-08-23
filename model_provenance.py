"""Generic checkpoint provenance contracts and persistent local cache.

Provenance answers *where this exact checkpoint came from* without making model
names part of runtime routing. Setup may use trusted remote providers to resolve
provenance by a strong content fingerprint. Doctor/runtime paths only consume the
persisted local record and never require network access.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping


CACHE_SCHEMA = 1
_HASH_CHUNK = 8 * 1024 * 1024


def provenance_cache_root() -> Path:
    raw = str(os.getenv("DUCKMOTION_PROVENANCE_CACHE") or "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".cache" / "duckmotion" / "provenance"


def _index_path() -> Path:
    return provenance_cache_root() / "index.json"


def _path_key(path: str | Path) -> str:
    return str(Path(path).expanduser().absolute())


def _stat_state(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _load_index() -> dict[str, Any]:
    path = _index_path()
    if not path.exists():
        return {"schema": CACHE_SCHEMA, "fingerprints": {}, "records": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        value = {}
    if not isinstance(value, dict) or value.get("schema") != CACHE_SCHEMA:
        value = {}
    fingerprints = value.get("fingerprints") if isinstance(value.get("fingerprints"), dict) else {}
    records = value.get("records") if isinstance(value.get("records"), dict) else {}
    return {"schema": CACHE_SCHEMA, "fingerprints": fingerprints, "records": records}


def _save_index(value: Mapping[str, Any]) -> None:
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def checkpoint_fingerprint(checkpoint: str | Path) -> dict[str, Any]:
    """Return a cached-or-computed SHA256 tied to the file's size and mtime."""
    path = Path(checkpoint).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    state = _stat_state(path)
    key = _path_key(path)
    index = _load_index()
    cached = index["fingerprints"].get(key)
    if (
        isinstance(cached, dict)
        and int(cached.get("size", -1)) == state["size"]
        and int(cached.get("mtime_ns", -1)) == state["mtime_ns"]
        and len(str(cached.get("sha256") or "")) == 64
    ):
        return {**state, "sha256": str(cached["sha256"]).lower(), "cached": True}

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    result = {**state, "sha256": digest.hexdigest(), "cached": False}
    index["fingerprints"][key] = {**state, "sha256": result["sha256"]}
    # A changed file invalidates any provenance record previously associated with
    # this filesystem path.
    index["records"].pop(key, None)
    _save_index(index)
    return result


@dataclass(frozen=True)
class CheckpointProvenanceProvider:
    provider_id: str
    resolve: Callable[[Path, Mapping[str, Any], Path], dict[str, Any] | None]


class CheckpointProvenanceRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, CheckpointProvenanceProvider] = {}

    def register(self, provider: CheckpointProvenanceProvider) -> None:
        provider_id = str(provider.provider_id or "").strip()
        if not provider_id:
            raise ValueError("Checkpoint provenance provider must define provider_id")
        if provider_id in self._providers:
            raise ValueError(f"Duplicate DuckMotion provenance provider: {provider_id}")
        self._providers[provider_id] = provider

    def providers(self) -> tuple[CheckpointProvenanceProvider, ...]:
        return tuple(self._providers.values())

    def ids(self) -> tuple[str, ...]:
        return tuple(self._providers.keys())


checkpoint_provenance_providers = CheckpointProvenanceRegistry()


def _record_matches(path: Path, record: Mapping[str, Any]) -> bool:
    try:
        state = _stat_state(path)
    except OSError:
        return False
    return (
        int(record.get("size", -1)) == state["size"]
        and int(record.get("mtime_ns", -1)) == state["mtime_ns"]
        and len(str(record.get("sha256") or "")) == 64
    )


def cached_provenance(checkpoint: str | Path) -> dict[str, Any] | None:
    """Read provenance for an unchanged local checkpoint without hashing/network."""
    path = Path(checkpoint).expanduser()
    if not path.exists() or not path.is_file():
        return None
    record = _load_index()["records"].get(_path_key(path))
    if not isinstance(record, dict) or not _record_matches(path, record):
        return None
    return dict(record)


def cached_recipe_paths(checkpoint: str | Path) -> tuple[Path, ...]:
    record = cached_provenance(checkpoint)
    if not record:
        return ()
    paths: list[Path] = []
    for raw in record.get("recipe_paths") or []:
        path = Path(str(raw)).expanduser()
        if path.exists() and path.is_file() and path.suffix.lower() == ".json":
            paths.append(path)
    return tuple(paths)


def prepare_checkpoint_provenance(checkpoint: str | Path) -> dict[str, Any]:
    """Resolve trusted provenance once and persist it for offline runtime use.

    Providers are queried with the strong SHA256 fingerprint. Exactly one provider
    must match; ambiguous matches are deliberately not cached.
    """
    path = Path(checkpoint).expanduser()
    fingerprint = checkpoint_fingerprint(path)
    cache_root = provenance_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)

    matches: list[dict[str, Any]] = []
    errors: list[str] = []
    for provider in checkpoint_provenance_providers.providers():
        try:
            result = provider.resolve(path, fingerprint, cache_root)
        except Exception as exc:
            errors.append(f"{provider.provider_id}: {exc}")
            continue
        if not result:
            continue
        matches.append({"provider": provider.provider_id, **dict(result)})

    if len(matches) != 1:
        return {
            "matched": False,
            "ambiguous": len(matches) > 1,
            "sha256": fingerprint["sha256"],
            "matches": matches,
            "errors": errors,
        }

    match = matches[0]
    recipe_paths = [
        str(Path(raw).expanduser().absolute())
        for raw in match.get("recipe_paths") or []
        if str(raw).strip()
    ]
    record = {
        "provider": match.get("provider"),
        "source_id": match.get("source_id"),
        "source_url": match.get("source_url"),
        "model_name": match.get("model_name"),
        "version_name": match.get("version_name"),
        "sha256": fingerprint["sha256"],
        "size": fingerprint["size"],
        "mtime_ns": fingerprint["mtime_ns"],
        "recipe_paths": recipe_paths,
        "metadata_path": match.get("metadata_path"),
    }
    index = _load_index()
    index["records"][_path_key(path)] = record
    _save_index(index)
    return {"matched": True, "record": record, "errors": errors}
