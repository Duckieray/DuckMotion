#!/usr/bin/env python3
"""Install DuckMotion as a WebbDuck web-app plugin."""

from __future__ import annotations

import argparse
import hashlib
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
PLUGIN_ID = "duckmotion"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_plugins_root() -> Path | None:
    raw = str(os.environ.get("WEBBDUCK_PLUGINS_DIR") or "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _webbduck_local_plugins_root(args: argparse.Namespace) -> Path | None:
    if not args.webbduck_dir:
        return None
    return (Path(args.webbduck_dir).expanduser().resolve() / "plugins").resolve()


def _resolve_plugins_root(args: argparse.Namespace) -> tuple[Path, str]:
    """Resolve the installation root using WebbDuck's actual discovery priority.

    An explicit ``--plugins-dir`` remains an exact operator override.  When the
    convenience ``--webbduck-dir`` argument is used, however, an existing
    ``WEBBDUCK_PLUGINS_DIR`` must win because WebbDuck itself searches that root
    first.  Installing into the repo-local root while an env root shadows it is
    indistinguishable from a successful-but-ignored update to a normal user.
    """

    if args.plugins_dir:
        return Path(args.plugins_dir).expanduser().resolve(), "--plugins-dir"

    env_root = _env_plugins_root()
    local_root = _webbduck_local_plugins_root(args)
    if env_root is not None:
        if local_root is not None and env_root != local_root:
            return env_root, "WEBBDUCK_PLUGINS_DIR (higher priority than --webbduck-dir)"
        return env_root, "WEBBDUCK_PLUGINS_DIR"

    if local_root is not None:
        return local_root, "--webbduck-dir"

    return (Path.home() / ".webbduck" / "plugins").resolve(), "~/.webbduck/plugins"


def _webbduck_search_roots(args: argparse.Namespace) -> list[tuple[Path, str]]:
    """Return the roots WebbDuck can search, in effective priority order."""

    rows: list[tuple[Path, str]] = []
    env_root = _env_plugins_root()
    local_root = _webbduck_local_plugins_root(args)
    home_root = (Path.home() / ".webbduck" / "plugins").resolve()

    if env_root is not None:
        rows.append((env_root, "WEBBDUCK_PLUGINS_DIR"))
    if local_root is not None:
        rows.append((local_root, "WebbDuck repo"))
    rows.append((home_root, "user fallback"))

    deduped: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for root, source in rows:
        if root in seen:
            continue
        seen.add(root)
        deduped.append((root, source))
    return deduped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install DuckMotion as a WebbDuck web-app plugin.",
    )
    parser.add_argument(
        "--plugins-dir",
        default=None,
        help="Exact WebbDuck plugins root (contains webapps/ and captioners/).",
    )
    parser.add_argument(
        "--webbduck-dir",
        default=None,
        help=(
            "Path to WebbDuck repo root. If WEBBDUCK_PLUGINS_DIR is set, the "
            "higher-priority env root is updated instead so WebbDuck actually "
            "loads the new plugin."
        ),
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
    """Return DuckMotion's root-level runtime modules."""

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

    for src in _runtime_python_files(source_root):
        _copy_path(src, target_dir / src.name)

    for rel in STATIC_INCLUDE_PATHS:
        src = source_root / rel
        if src.exists():
            _copy_path(src, target_dir / rel)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_ui_copy(source_root: Path, target_dir: Path) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for relative in (Path("ui/index.html"), Path("ui/app.js")):
        source = source_root / relative
        target = target_dir / relative
        if not source.exists() or not target.exists() or _sha256(source) != _sha256(target):
            mismatches.append(str(relative))
    return not mismatches, mismatches


def _duplicate_plugin_roots(args: argparse.Namespace, selected_root: Path) -> list[tuple[Path, str]]:
    duplicates: list[tuple[Path, str]] = []
    for root, source in _webbduck_search_roots(args):
        if root == selected_root:
            continue
        if (root / "webapps" / PLUGIN_ID / "plugin.json").exists():
            duplicates.append((root, source))
    return duplicates


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

    plugins_root, root_reason = _resolve_plugins_root(args)
    target_dir = plugins_root / "webapps" / PLUGIN_ID
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

    ui_ok, ui_mismatches = _verify_ui_copy(source_root, target_dir)
    if not ui_ok:
        print(
            "ERROR: installed DuckMotion UI does not match the source checkout: "
            + ", ".join(ui_mismatches),
            file=sys.stderr,
        )
        return 4

    duplicates = _duplicate_plugin_roots(args, plugins_root)

    print("DuckMotion WebbDuck plugin installed.")
    print(f"source: {source_root}")
    print(f"target: {target_dir}")
    print(f"selected root: {root_reason}")
    print(f"backend: {backend_name}")
    print("ui verification: source and installed index.html/app.js match")
    if duplicates:
        print("")
        print("NOTE: other DuckMotion copies also exist, but do not shadow this install:")
        for root, source in duplicates:
            print(f" - {root / 'webapps' / PLUGIN_ID} ({source})")
    print("")
    print("Next:")
    print("1) Restart WebbDuck so its backend module and plugin discovery reload.")
    print("2) Reload the WebbDuck page and select the DuckMotion tab.")
    print("3) Run DuckMotion runtime readiness before the first generation smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
