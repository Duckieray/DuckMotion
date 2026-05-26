#!/usr/bin/env python3
"""Install DuckMotion as a WebbDuck web-app plugin."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_plugins_root(args: argparse.Namespace) -> Path:
    if args.plugins_dir:
        return Path(args.plugins_dir).expanduser().resolve()

    if args.webbduck_dir:
        return (Path(args.webbduck_dir).expanduser().resolve() / "plugins").resolve()

    env_dir = os.environ.get("WEBBDUCK_PLUGINS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    return (Path.home() / ".webbduck" / "plugins").resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install DuckMotion as a WebbDuck web-app plugin.",
    )
    parser.add_argument(
        "--plugins-dir",
        default=None,
        help="WebbDuck plugins root (contains webapps/ and captioners/).",
    )
    parser.add_argument(
        "--webbduck-dir",
        default=None,
        help="Path to WebbDuck repo root (installs into <webbduck-dir>/plugins).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing plugin files if already installed.",
    )
    return parser.parse_args()


def _copy_plugin_tree(source_root: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    include_paths = ("plugin.json", "backend.py", "ui", "README.md")
    for rel in include_paths:
        src = source_root / rel
        if not src.exists():
            continue
        dst = target_dir / rel
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main() -> int:
    args = _parse_args()
    source_root = _repo_root()
    required = [source_root / "plugin.json", source_root / "backend.py", source_root / "ui"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("ERROR: missing required plugin files:", file=sys.stderr)
        for row in missing:
            print(f" - {row}", file=sys.stderr)
        return 1

    plugins_root = _resolve_plugins_root(args)
    target_dir = plugins_root / "webapps" / "duckmotion"
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if target_dir.exists() and not args.overwrite:
        print(
            "ERROR: target already exists. Re-run with --overwrite to replace.\n"
            f"target={target_dir}",
            file=sys.stderr,
        )
        return 2

    if target_dir.exists():
        shutil.rmtree(target_dir)

    _copy_plugin_tree(source_root, target_dir)

    print("DuckMotion WebbDuck plugin installed.")
    print(f"source: {source_root}")
    print(f"target: {target_dir}")
    print("")
    print("Next:")
    print("1) Start/restart WebbDuck.")
    print("2) Open WebbDuck and select the DuckMotion tab.")
    print("3) Configure DuckMotion model/runtime settings in DuckMotion Setup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
