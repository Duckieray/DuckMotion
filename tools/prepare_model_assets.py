#!/usr/bin/env python3
"""Prepare recipe-declared support assets for discovered DuckMotion models.

This tool is architecture- and model-name agnostic. Registered asset providers
discover checkpoints and normalize their recipe-declared assets. When a local
recipe is absent, generic provenance providers may identify the exact checkpoint
by strong content hash and cache trusted recipe sidecars for offline use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_asset_providers import ModelAssetProvider, model_asset_providers
from model_provenance import prepare_checkpoint_provenance
from model_recipes import execution_profiles
from provenance_providers import register_builtin_provenance_providers
from runtime_paths import resolve_runtime_python


HF_DOWNLOAD_SCRIPT = r'''
import json, sys
from huggingface_hub import hf_hub_download
repo_id, filename, revision, local_dir = sys.argv[1:5]
try:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision or None,
        local_dir=local_dir,
    )
    print(json.dumps({"ok": True, "path": path}))
except Exception as exc:
    print(json.dumps({"ok": False, "type": exc.__class__.__name__, "error": str(exc)}))
    raise SystemExit(2)
'''


def _hf_source(url: str) -> tuple[str, str, str] | None:
    """Parse a normal huggingface.co /resolve/ asset URL."""
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if parsed.scheme != "https" or parsed.netloc.lower() not in {
        "huggingface.co",
        "www.huggingface.co",
    }:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 5 or parts[2] != "resolve":
        return None
    repo_id = f"{parts[0]}/{parts[1]}"
    revision = parts[3]
    filename = "/".join(parts[4:])
    if not repo_id or not filename:
        return None
    return repo_id, filename, revision


def _download_hf(
    runtime_python: str,
    *,
    url: str,
    cache: Path,
) -> tuple[bool, str]:
    source = _hf_source(url)
    if source is None:
        return False, "recipe asset does not declare a supported trusted Hugging Face source URL"
    repo_id, filename, revision = source
    completed = subprocess.run(
        [
            runtime_python,
            "-c",
            HF_DOWNLOAD_SCRIPT,
            repo_id,
            filename,
            revision,
            str(cache),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = (completed.stdout or "").strip().splitlines()
    payload = {}
    if lines:
        try:
            payload = json.loads(lines[-1])
        except Exception:
            payload = {}
    if completed.returncode == 0 and payload.get("ok"):
        return True, str(payload.get("path") or "")
    detail = str(
        payload.get("error")
        or (completed.stderr or "").strip()
        or "download failed"
    )
    return False, detail


def _provenance_message(checkpoint: Path, result: dict) -> str:
    if result.get("matched"):
        record = dict(result.get("record") or {})
        source = str(record.get("source_id") or record.get("source_url") or "trusted provider")
        recipes = record.get("recipe_paths") or []
        if recipes:
            cache_note = "cached" if result.get("cached") else "cached now"
            return f"{checkpoint.name}: provenance matched {source}; {cache_note} {len(recipes)} recipe candidate(s)"
        diagnostics = record.get("diagnostics") if isinstance(record.get("diagnostics"), dict) else {}
        candidates = int(diagnostics.get("recipe_candidates") or 0)
        errors = diagnostics.get("recipe_errors") or []
        if candidates:
            detail = f": {errors[0]}" if errors else ""
            return f"{checkpoint.name}: provenance matched {source}; {candidates} JSON sidecar(s) were listed but not cached{detail}"
        return f"{checkpoint.name}: provenance matched {source}, but it publishes no JSON recipe sidecar"
    if result.get("ambiguous"):
        return f"{checkpoint.name}: checkpoint provenance is ambiguous across trusted providers"
    errors = result.get("errors") or []
    if errors:
        return f"{checkpoint.name}: provenance lookup unavailable: {errors[0]}"
    return f"{checkpoint.name}: no trusted checkpoint provenance match was found"


def _asset_declaration(
    profile_id: str,
    kind: str,
    declaration: dict,
) -> tuple[str, str]:
    """Merge trusted profile sources without substituting custom recipe assets."""
    name = str((declaration or {}).get("name") or "").strip()
    url = str((declaration or {}).get("url") or "").strip()
    profile = execution_profiles.get(profile_id)
    fallback = (
        dict(profile.asset_defaults.get(kind) or {})
        if profile is not None and isinstance(profile.asset_defaults, dict)
        else {}
    )
    fallback_name = str(fallback.get("name") or "").strip()
    fallback_url = str(fallback.get("url") or "").strip()
    if not name and fallback_name:
        name = fallback_name
    if not url and name and fallback_name and name.lower() == fallback_name.lower():
        url = fallback_url
    return name, url


def _prepare_provider(
    provider: ModelAssetProvider,
    models_dir: Path,
) -> tuple[int, list[str]]:
    checkpoints = provider.discover(models_dir)
    if not checkpoints:
        return 0, []

    runtime_python = str(resolve_runtime_python(provider.runtime_id))
    if not Path(runtime_python).expanduser().exists():
        return 2, [f"runtime is missing: {runtime_python}"]

    cache = provider.cache_root().expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []

    for checkpoint in checkpoints:
        state = provider.inspect(checkpoint, models_dir=str(models_dir))
        profile = str(state.get("execution_profile") or "").strip()
        provenance_result: dict | None = None
        if not profile:
            print(f"  identifying {checkpoint.name} by SHA256 provenance (cached after first run)")
            provenance_result = prepare_checkpoint_provenance(checkpoint)
            print(f"    {_provenance_message(checkpoint, provenance_result)}")
            # Provenance providers only cache candidate JSON. The model-specific
            # recipe adapter remains authoritative about whether any candidate
            # actually maps to an installed execution profile.
            state = provider.inspect(checkpoint, models_dir=str(models_dir))
            profile = str(state.get("execution_profile") or "").strip()

        if not profile:
            if provenance_result and provenance_result.get("matched"):
                record = dict(provenance_result.get("record") or {})
                if record.get("recipe_paths"):
                    blockers.append(
                        f"{checkpoint.name}: trusted provenance was resolved, but none of its recipe sidecars map to an installed execution profile"
                    )
                else:
                    diagnostics = record.get("diagnostics") if isinstance(record.get("diagnostics"), dict) else {}
                    candidates = int(diagnostics.get("recipe_candidates") or 0)
                    errors = diagnostics.get("recipe_errors") or []
                    if candidates:
                        detail = f": {errors[0]}" if errors else ""
                        blockers.append(
                            f"{checkpoint.name}: provenance listed {candidates} recipe sidecar(s), but none could be cached{detail}"
                        )
                    else:
                        blockers.append(
                            f"{checkpoint.name}: trusted provenance was resolved, but no recipe sidecar is published for this version"
                        )
            else:
                blockers.append(
                    f"{checkpoint.name}: format recognized, but no supported execution recipe was resolved"
                )
            continue

        manifest = dict(state.get("asset_manifest") or {})
        resolved = dict(state.get("assets") or {})
        profile_obj = execution_profiles.get(profile)
        asset_roles = profile_obj.required_assets if profile_obj is not None else tuple(manifest.keys())
        for kind in asset_roles:
            if resolved.get(kind):
                continue
            declaration = dict(manifest.get(kind) or {})
            name, url = _asset_declaration(profile, kind, declaration)
            if not name:
                blockers.append(
                    f"{checkpoint.name}: recipe '{profile}' does not declare required asset role '{kind}' and the profile has no default"
                )
                continue
            if not url:
                blockers.append(
                    f"{checkpoint.name}: {name} has no trusted source URL in the recipe/profile"
                )
                continue
            print(f"  fetching {name}")
            ok, detail = _download_hf(runtime_python, url=url, cache=cache)
            if ok:
                print(f"    ✓ {detail}")
            else:
                blockers.append(f"{checkpoint.name}: {name}: {detail}")
                print(f"    ✗ {detail}")

    return (3 if blockers else 0), blockers


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare checkpoint provenance and recipe-declared support assets for DuckMotion models."
    )
    parser.add_argument(
        "--models",
        required=True,
        type=Path,
        help="Shared model-library root.",
    )
    args = parser.parse_args()

    models_dir = args.models.expanduser().resolve()
    if not models_dir.exists() or not models_dir.is_dir():
        parser.error(f"Model directory does not exist: {models_dir}")

    register_builtin_provenance_providers()
    print("Preparing model provenance and declared support assets")
    blockers: list[str] = []
    status = 0
    for provider in model_asset_providers.providers():
        code, messages = _prepare_provider(provider, models_dir)
        if messages:
            print(f"{provider.label}:")
            for message in messages:
                print(f"  ! {message}")
        blockers.extend(messages)
        status = max(status, code)

    if blockers:
        print("")
        print("Some model recipes/assets are not ready.")
        print(
            "DuckMotion does not require a special folder layout; resolve the reported "
            "provenance/recipe/access issue and rerun setup."
        )
        print(
            "For gated Hugging Face assets, accept the repository terms and "
            "authenticate normally."
        )
        return status or 3

    print("Model provenance and declared support assets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
