#!/usr/bin/env python3
"""Install DuckMotion as a WebbDuck web-app plugin."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


STATIC_INCLUDE_PATHS = (
    "plugin.json",
    "ui",
    "runtime_requirements",
    "requirements.txt",
    "README.md",
)


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


def _load_manifest(source_root: Path) -> dict:
    manifest_path = source_root / "plugin.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Unable to read plugin manifest: {manifest_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Plugin manifest must contain a JSON object: {manifest_path}")
    return value


def _runtime_python_files(source_root: Path) -> list[Path]:
    """Return DuckMotion's root-level runtime modules.

    Runtime modules intentionally live at the plugin root so WebbDuck can load
    the manifest backend as a standalone web plugin. Tests, tools and docs live
    in subdirectories and are therefore not copied by this rule.
    """
    return sorted(
        (path for path in source_root.glob("*.py") if path.is_file()),
        key=lambda path: path.name.lower(),
    )


def _required_paths(source_root: Path, manifest: dict) -> list[Path]:
    backend_name = str(manifest.get("backend") or "plugin_backend.py").strip()
    required = [source_root / "plugin.json", source_root / "ui"]
    if backend_name:
        required.append(source_root / backend_name)
    return required


def _copy_path(src: Path, dst: Path) -> None:
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


def _copy_plugin_tree(source_root: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy every root-level Python module. The modular runtime is deliberately
    # composed from sibling modules (model discovery/runtime, storage, Wan/LTX
    # adapters/workers, readiness helpers, etc.); packaging only the manifest
    # backend would leave a plugin that imports successfully only until its
    # first sibling import.
    for src in _runtime_python_files(source_root):
        _copy_path(src, target_dir / src.name)

    for rel in STATIC_INCLUDE_PATHS:
        src = source_root / rel
        if src.exists():
            _copy_path(src, target_dir / rel)


def main() -> int:
    args = _parse_args()
    source_root = _repo_root()

    try:
        manifest = _load_manifest(source_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    required = _required_paths(source_root, manifest)
    missing = [str(path) for path in required if not path.exists()]
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

    backend_name = str(manifest.get("backend") or "plugin_backend.py").strip()
    installed_backend = target_dir / backend_name
    if backend_name and not installed_backend.exists():
        print(f"ERROR: installed plugin is missing manifest backend: {installed_backend}", file=sys.stderr)
        return 3

    print("DuckMotion WebbDuck plugin installed.")
    print(f"source: {source_root}")
    print(f"target: {target_dir}")
    print(f"backend: {backend_name}")
    print("")
    print("Next:")
    print("1) Start/restart WebbDuck.")
    print("2) Open WebbDuck and select the DuckMotion tab.")
    print("3) Run DuckMotion runtime readiness before the first generation smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
