#!/usr/bin/env python3
"""Prepare declared support assets for discovered DuckMotion models.

This is intentionally model-name agnostic. Providers inspect architecture/format
and declarative recipe metadata, then fetch only assets explicitly declared by
that recipe. Brand names such as REDGraft are never used to select behavior.

Today the first provider handles LTX-2.5 INT8 ConvRot companion recipes. The
provider boundary is deliberately generic so Wan/LTX/etc. can add their own
asset preparers without changing setup.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltx_convrot_assets import convrot_asset_cache_root, inspect_convrot_assets, is_ltx25_convrot_path
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
    if parsed.scheme != "https" or parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
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


def _download_hf(runtime_python: str, *, url: str, cache: Path) -> tuple[bool, str]:
    source = _hf_source(url)
    if source is None:
        return False, "recipe asset does not declare a supported trusted Hugging Face source URL"
    repo_id, filename, revision = source
    completed = subprocess.run(
        [runtime_python, "-c", HF_DOWNLOAD_SCRIPT, repo_id, filename, revision, str(cache)],
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
    detail = str(payload.get("error") or (completed.stderr or "").strip() or "download failed")
    return False, detail


def _discover_convrot(models_dir: Path) -> list[Path]:
    try:
        candidates = sorted(models_dir.rglob("*.safetensors"), key=lambda p: str(p).lower())
    except OSError:
        return []
    return [path for path in candidates if is_ltx25_convrot_path(path)]


def _prepare_ltx25_convrot(models_dir: Path) -> tuple[int, list[str]]:
    checkpoints = _discover_convrot(models_dir)
    if not checkpoints:
        return 0, []

    runtime_python = str(resolve_runtime_python("ltx25_convrot"))
    if not Path(runtime_python).expanduser().exists():
        return 2, [f"LTX ConvRot runtime is missing: {runtime_python}"]

    cache = convrot_asset_cache_root().expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []

    for checkpoint in checkpoints:
        state = inspect_convrot_assets(checkpoint, models_dir=str(models_dir))
        profile = state.get("execution_profile")
        if not profile:
            blockers.append(f"{checkpoint.name}: no supported companion execution recipe was found")
            continue

        manifest = dict(state.get("asset_manifest") or {})
        resolved = dict(state.get("assets") or {})
        for kind, declaration in manifest.items():
            if resolved.get(kind):
                continue
            name = str((declaration or {}).get("name") or "").strip()
            url = str((declaration or {}).get("url") or "").strip()
            if not name:
                blockers.append(f"{checkpoint.name}: recipe does not declare required {kind}")
                continue
            if not url:
                blockers.append(f"{checkpoint.name}: {name} has no trusted source URL in the recipe")
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
    parser = argparse.ArgumentParser(description="Prepare declared support assets for discovered DuckMotion models.")
    parser.add_argument("--models", required=True, type=Path, help="Shared model-library root.")
    args = parser.parse_args()

    models_dir = args.models.expanduser().resolve()
    if not models_dir.exists() or not models_dir.is_dir():
        parser.error(f"Model directory does not exist: {models_dir}")

    print("Preparing model-declared support assets")
    providers = [
        ("LTX-2.5 ConvRot", _prepare_ltx25_convrot),
    ]
    blockers: list[str] = []
    status = 0
    for label, provider in providers:
        code, messages = provider(models_dir)
        if messages:
            print(f"{label}:")
            for message in messages:
                print(f"  ! {message}")
        blockers.extend(messages)
        status = max(status, code)

    if blockers:
        print("")
        print("Some model-declared assets are not ready.")
        print("DuckMotion does not require a special folder layout; fix the reported recipe/access issue and rerun setup.")
        print("For gated Hugging Face assets, accept the repository terms and authenticate normally.")
        return status or 3

    print("Model-declared support assets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
