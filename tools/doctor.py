#!/usr/bin/env python3
"""Human-friendly DuckMotion installation/model readiness report.

This command does not load model weights or submit generation jobs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_paths import RUNTIME_ENV_VARS, configure_default_runtime_env, runtime_status

configure_default_runtime_env()

from ltx_backend import ensure_registered as ensure_ltx_registered
from ltx_convrot_backend import ensure_registered as ensure_ltx_convrot_registered
from model_discovery import discover_video_models
from model_runtime import backend_resolver, describe_video_model
from wan_backend import ensure_registered as ensure_wan_registered


CONFIG_FILE = Path.home() / ".webbduck" / "plugin_state" / "duckmotion_config.json"


def _load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {"model_id_or_path": "", "models_dir": "", "output_dir": ""}
    try:
        value = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"model_id_or_path": "", "models_dir": "", "output_dir": ""}
    if not isinstance(value, dict):
        value = {}
    return {
        "model_id_or_path": str(value.get("model_id_or_path") or "").strip(),
        "models_dir": str(value.get("models_dir") or "").strip(),
        "output_dir": str(value.get("output_dir") or "").strip(),
    }


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def _short_reason(payload: dict) -> str:
    reason = str(payload.get("reason") or "").strip()
    if reason:
        return reason
    return "ready" if payload.get("ready") else "not ready"


def main() -> int:
    config = _load_config()
    models_dir = str(config.get("models_dir") or "").strip()

    print("DuckMotion doctor")
    print("=================")
    print(f"Config: {CONFIG_FILE}")
    if models_dir:
        path = Path(models_dir).expanduser()
        print(f"Models: {_mark(path.exists() and path.is_dir())} {path}")
    else:
        print("Models: ✗ not configured (run: python tools/setup.py --models /path/to/models)")

    print("")
    print("Runtimes")
    runtime_ok = True
    for runtime in RUNTIME_ENV_VARS:
        status = runtime_status(runtime)
        exists = bool(status["exists"])
        runtime_ok = runtime_ok and exists
        print(
            f"  {_mark(exists)} {runtime:<13} {status['python']}"
            + ("" if status["source"] == "default" else " (override)")
        )

    ensure_wan_registered()
    ensure_ltx_registered()
    ensure_ltx_convrot_registered()

    catalog = discover_video_models(config)
    items = [item for item in catalog.get("items") or [] if isinstance(item, dict)]
    print("")
    print(f"Models discovered: {len(items)}")
    print(f"Hugging Face cache: {catalog.get('hf_cache')}")

    ready_count = 0
    for item in items:
        source = str(item.get("source") or item.get("repo_id") or item.get("name") or "").strip()
        name = str(item.get("name") or Path(source).name or source).strip()
        descriptor = describe_video_model(source, name=name)
        if not descriptor.supported:
            print(f"  ! {name}: detected, runtime not implemented")
            continue
        try:
            readiness = backend_resolver.readiness(descriptor)
        except Exception as exc:
            readiness = {"ready": False, "reason": str(exc or exc.__class__.__name__)}
        ready = bool(readiness.get("ready"))
        if ready:
            ready_count += 1
        print(f"  {_mark(ready)} {name}: {_short_reason(readiness)}")
        assets = readiness.get("assets") if isinstance(readiness.get("assets"), dict) else None
        if assets and not assets.get("ready"):
            for missing in assets.get("missing") or []:
                print(f"      missing: {missing}")

    print("")
    if not items:
        print("No DuckMotion video models were discovered.")
    elif ready_count:
        print(f"Ready models: {ready_count}/{len(items)}")
    else:
        print("No discovered model is generation-ready yet.")

    if not runtime_ok:
        print("Run: python tools/setup.py --models /path/to/models")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
