"""Trusted Civitai checkpoint provenance provider.

The public Civitai model-version API supports lookup by full SHA256. DuckMotion
uses that strong content identity only during setup, caches the returned version
metadata locally, and downloads only small JSON sidecars from the matched public
model version. Model weights are never re-downloaded by this provider.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping
import urllib.error
import urllib.request
from urllib.parse import urlparse

from provider_credentials import provider_token


API_BY_HASH = "https://civitai.com/api/v1/model-versions/by-hash/{sha256}"
DOWNLOAD_BY_VERSION = "https://civitai.com/api/download/models/{version_id}?fileId={file_id}"
MAX_API_BYTES = 8 * 1024 * 1024
MAX_RECIPE_BYTES = 32 * 1024 * 1024
USER_AGENT = "DuckMotion/1 checkpoint-provenance"


def _trusted_civitai_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    return parsed.scheme == "https" and parsed.netloc.lower() in {
        "civitai.com",
        "www.civitai.com",
    }


def _civitai_token() -> str:
    return provider_token("civitai")


def _headers(accept: str) -> dict[str, str]:
    headers = {"Accept": accept, "User-Agent": USER_AGENT}
    token = _civitai_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _read_limited(response, limit: int) -> bytes:
    payload = response.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"remote payload exceeds {limit} bytes")
    return payload


def _get_json(url: str, *, limit: int) -> Any:
    request = urllib.request.Request(url, headers=_headers("application/json"))
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = _read_limited(response, limit)
    return json.loads(raw.decode("utf-8"))


def _safe_filename(name: str, fallback: str) -> str:
    value = Path(str(name or "")).name
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value or fallback


def _file_download_url(item: Mapping[str, Any], version_id: Any) -> str:
    """Return a file-specific Civitai download URL when identity is available.

    Civitai model-version responses historically exposed version-level download
    URLs that could be re-resolved to a different primary file. Current Civitai
    code pins per-file URLs with ``?fileId=<id>``. Build that canonical URL from
    the response's version/file identities instead of trusting a possibly stale
    or unpinned ``downloadUrl`` field.
    """
    try:
        normalized_version_id = int(version_id)
        normalized_file_id = int(item.get("id"))
    except (TypeError, ValueError):
        normalized_version_id = 0
        normalized_file_id = 0

    if normalized_version_id > 0 and normalized_file_id > 0:
        return DOWNLOAD_BY_VERSION.format(
            version_id=normalized_version_id,
            file_id=normalized_file_id,
        )

    published = str(item.get("downloadUrl") or "").strip()
    return published if published and _trusted_civitai_url(published) else ""


def _download_recipe(url: str, destination: Path) -> Path:
    if not _trusted_civitai_url(url):
        raise ValueError("Civitai recipe file declares an untrusted download URL")
    request = urllib.request.Request(url, headers=_headers("application/json,*/*"))
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = _read_limited(response, MAX_RECIPE_BYTES)
    # A provenance sidecar can be any exported-workflow object, but it must be
    # valid JSON before it enters DuckMotion's declarative recipe adapters.
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Civitai JSON sidecar is not a JSON object")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    temp.replace(destination)
    return destination


def resolve_civitai_provenance(
    checkpoint: Path,
    fingerprint: Mapping[str, Any],
    cache_root: Path,
) -> dict[str, Any] | None:
    sha256 = str(fingerprint.get("sha256") or "").lower()
    if len(sha256) != 64:
        raise ValueError("Civitai provenance requires a full SHA256 fingerprint")

    url = API_BY_HASH.format(sha256=sha256.upper())
    try:
        payload = _get_json(url, limit=MAX_API_BYTES)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise RuntimeError(f"Civitai lookup returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Civitai lookup failed: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Civitai provenance response is not a JSON object")

    version_id = payload.get("id")
    model_id = payload.get("modelId")
    files = payload.get("files") if isinstance(payload.get("files"), list) else []
    # Verify that the version response actually contains this exact SHA256 before
    # trusting any companion data from it.
    exact_match = False
    for item in files:
        if not isinstance(item, dict):
            continue
        hashes = item.get("hashes") if isinstance(item.get("hashes"), dict) else {}
        if str(hashes.get("SHA256") or "").lower() == sha256:
            exact_match = True
            break
    if not exact_match:
        raise ValueError("Civitai lookup did not echo an exact matching SHA256 file")

    provider_root = cache_root / "providers" / "civitai" / sha256
    provider_root.mkdir(parents=True, exist_ok=True)
    metadata_path = provider_root / "model-version.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    recipe_paths: list[str] = []
    recipe_errors: list[str] = []
    recipe_candidates = 0
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if not name.lower().endswith(".json"):
            continue
        recipe_candidates += 1
        size_kb = item.get("sizeKB")
        try:
            if size_kb is not None and float(size_kb) * 1024 > MAX_RECIPE_BYTES:
                recipe_errors.append(f"{name}: file is larger than the JSON sidecar safety limit")
                continue
        except (TypeError, ValueError):
            pass
        download_url = _file_download_url(item, version_id)
        if not download_url:
            recipe_errors.append(f"{name}: no trustworthy file-specific download URL could be derived")
            continue
        destination = provider_root / _safe_filename(name, f"recipe-{index}.json")
        try:
            recipe_paths.append(str(_download_recipe(download_url, destination)))
        except Exception as exc:
            # One malformed/unavailable sidecar must not erase otherwise useful
            # provenance. Recipe adapters will evaluate the successfully cached
            # candidates below.
            recipe_errors.append(f"{name}: {exc}")

    model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    source_url = None
    if model_id and version_id:
        source_url = f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
    return {
        "source_id": f"civitai:{model_id}@{version_id}",
        "source_url": source_url,
        "model_name": model.get("name"),
        "version_name": payload.get("name"),
        "recipe_paths": recipe_paths,
        "metadata_path": str(metadata_path),
        "diagnostics": {
            "recipe_candidates": recipe_candidates,
            "recipe_errors": recipe_errors,
        },
    }
