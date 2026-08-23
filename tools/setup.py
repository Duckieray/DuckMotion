#!/usr/bin/env python3
"""One-command DuckMotion setup for normal users.

Examples:
    python tools/setup.py --models /path/to/models
    python tools/setup.py --models /path/to/models --webbduck-dir ../WebbDuck

The command prepares isolated runtimes, repairs runtime-owned adjuncts, persists
the shared model root, prepares support assets for discovered models, and
installs/refreshes the WebbDuck plugin when a checkout is available.
Environment variables and special model folder layouts are not required for the
normal path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = Path.home() / ".webbduck" / "plugin_state"
CONFIG_FILE = STATE_DIR / "duckmotion_config.json"


def _load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        value = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _save_models_dir(models_dir: Path) -> None:
    value = _load_config()
    value["models_dir"] = str(models_dir)
    value.setdefault("model_id_or_path", "")
    value.setdefault("output_dir", "")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _discover_webbduck(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.exists() else None

    candidates = [
        ROOT.parent / "WebbDuck",
        Path.cwd().parent / "WebbDuck",
    ]
    for candidate in candidates:
        if (candidate / "run.py").exists() or (candidate / "server").exists():
            return candidate.resolve()
    return None


def _run(args: list[str]) -> None:
    print("+", " ".join(args))
    subprocess.check_call(args)


def _run_nonfatal(args: list[str]) -> int:
    print("+", " ".join(args))
    return subprocess.run(args, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare DuckMotion for normal local use.")
    parser.add_argument(
        "--models",
        required=True,
        help="Shared model-library root. DuckMotion scans this plus the normal Hugging Face cache.",
    )
    parser.add_argument(
        "--webbduck-dir",
        default=None,
        help="Optional WebbDuck checkout. A sibling ../WebbDuck checkout is detected automatically.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Base Python used to create isolated runtimes.",
    )
    parser.add_argument(
        "--skip-runtimes",
        action="store_true",
        help="Keep installed runtime packages; runtime-owned adjuncts are still repaired.",
    )
    parser.add_argument(
        "--skip-model-assets",
        action="store_true",
        help="Do not prepare support assets for discovered models.",
    )
    parser.add_argument(
        "--skip-plugin-install",
        action="store_true",
        help="Do not install/refresh the WebbDuck plugin.",
    )
    args = parser.parse_args()

    models_dir = Path(args.models).expanduser().resolve()
    if not models_dir.exists() or not models_dir.is_dir():
        parser.error(f"Model directory does not exist: {models_dir}")

    print(f"Models: {models_dir}")
    _save_models_dir(models_dir)
    print(f"Saved DuckMotion config: {CONFIG_FILE}")

    runtime_tool = str(ROOT / "tools" / "prepare_model_runtimes.py")
    if args.skip_runtimes:
        # Skipping package installation must not leave runtime-owned resources
        # (notably the pinned ConvRot Comfy checkout) half-created.
        _run(
            [
                args.python,
                runtime_tool,
                "all",
                "--repair-only",
            ]
        )
    else:
        _run(
            [
                args.python,
                runtime_tool,
                "all",
                "--python",
                args.python,
            ]
        )

    asset_status = 0
    if not args.skip_model_assets:
        asset_status = _run_nonfatal(
            [
                args.python,
                str(ROOT / "tools" / "prepare_convrot_assets.py"),
                "--models",
                str(models_dir),
            ]
        )

    webbduck = _discover_webbduck(args.webbduck_dir)
    if not args.skip_plugin_install:
        if webbduck is None:
            print("WebbDuck checkout not found; runtime/model setup is complete.")
            print("If WebbDuck lives elsewhere, rerun with --webbduck-dir /path/to/WebbDuck.")
        else:
            _run(
                [
                    args.python,
                    str(ROOT / "tools" / "install_webbduck_plugin.py"),
                    "--webbduck-dir",
                    str(webbduck),
                    "--overwrite",
                ]
            )

    print("")
    if asset_status:
        print("Setup completed with one or more model-asset access blockers.")
        print("No manual folder arrangement is required; resolve the reported Hugging Face access issue and rerun setup.")
    else:
        print("Setup complete.")
    print("No DUCKMOTION_*_PYTHON exports are required.")
    print("Next: python tools/doctor.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
