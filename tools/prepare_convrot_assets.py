#!/usr/bin/env python3
"""Prepare support assets for discovered REDGraft LTX-2.5 ConvRot checkpoints.

The checkpoint itself is user-selected content and is never downloaded here.
Only the fixed support assets required by DuckMotion's audited REDGraft recipe
are fetched, and only when a matching checkpoint is already present.

Lightricks/LTX-2.5 is gated on Hugging Face. This helper respects normal
Hugging Face authentication/access and never attempts to bypass or accept model
terms on the user's behalf.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltx_convrot_assets import (
    REDGRAFT_ASSET_SOURCES,
    convrot_asset_cache_root,
    inspect_convrot_assets,
    is_redgraft_checkpoint,
)
from runtime_paths import resolve_runtime_python


DOWNLOAD_SCRIPT = r'''
import json, sys
from huggingface_hub import hf_hub_download
repo_id, filename, local_dir = sys.argv[1:4]
try:
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    print(json.dumps({"ok": True, "path": path}))
except Exception as exc:
    print(json.dumps({"ok": False, "type": exc.__class__.__name__, "error": str(exc)}))
    raise SystemExit(2)
'''


def discover_redgraft_checkpoints(models_dir: Path) -> list[Path]:
    try:
        candidates = sorted(models_dir.rglob("*.safetensors"), key=lambda p: str(p).lower())
    except OSError:
        return []
    return [path for path in candidates if is_redgraft_checkpoint(path)]


def _download(runtime_python: str, *, repo_id: str, filename: str, cache: Path) -> tuple[bool, str]:
    completed = subprocess.run(
        [runtime_python, "-c", DOWNLOAD_SCRIPT, repo_id, filename, str(cache)],
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare support assets for discovered REDGraft checkpoints.")
    parser.add_argument("--models", required=True, type=Path, help="Shared model-library root.")
    parser.add_argument(
        "--runtime-python",
        default=None,
        help="Advanced override for the ConvRot runtime interpreter.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=convrot_asset_cache_root(),
        help="DuckMotion-managed support-asset cache.",
    )
    args = parser.parse_args()

    models_dir = args.models.expanduser().resolve()
    if not models_dir.exists() or not models_dir.is_dir():
        parser.error(f"Model directory does not exist: {models_dir}")

    checkpoints = discover_redgraft_checkpoints(models_dir)
    if not checkpoints:
        print("No REDGraft LTX-2.5 ConvRot checkpoint discovered; no support assets needed.")
        return 0

    runtime_python = str(args.runtime_python or resolve_runtime_python("ltx25_convrot"))
    if not Path(runtime_python).expanduser().exists():
        print(f"ConvRot runtime is missing: {runtime_python}")
        print("Run DuckMotion setup normally to create the runtime first.")
        return 2

    cache = args.cache.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for checkpoint in checkpoints:
        print(f"REDGraft: {checkpoint}")
        state = inspect_convrot_assets(checkpoint, models_dir=str(models_dir))
        missing_names = {Path(value).name for value in state.get("missing") or [] if value and not str(value).startswith("<")}
        if not missing_names:
            print("  support assets already ready")
            continue

        for kind, spec in REDGRAFT_ASSET_SOURCES.items():
            name = spec["name"]
            if name not in missing_names:
                continue
            print(f"  fetching {name}")
            ok, detail = _download(
                runtime_python,
                repo_id=spec["repo_id"],
                filename=spec["filename"],
                cache=cache,
            )
            if ok:
                print(f"    ✓ {detail}")
            else:
                print(f"    ✗ {detail}")
                failures.append(f"{name}: {detail}")

    if failures:
        print("")
        print("Some REDGraft support assets could not be downloaded.")
        print("Lightricks/LTX-2.5 requires Hugging Face access. If the error is gated/401/403,")
        print("accept the repository terms and authenticate with Hugging Face, then rerun setup.")
        print("DuckMotion will keep using any assets that downloaded successfully.")
        return 3

    print(f"REDGraft support assets ready in: {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
