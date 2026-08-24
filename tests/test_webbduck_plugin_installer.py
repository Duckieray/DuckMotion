from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER_PATH = ROOT / "tools" / "install_webbduck_plugin.py"
SPEC = importlib.util.spec_from_file_location("duckmotion_install_webbduck_plugin", INSTALLER_PATH)
assert SPEC is not None and SPEC.loader is not None
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


def _args(*, plugins_dir=None, webbduck_dir=None):
    return argparse.Namespace(
        plugins_dir=plugins_dir,
        webbduck_dir=webbduck_dir,
        overwrite=True,
    )


def test_explicit_plugins_dir_is_exact_operator_override(tmp_path: Path, monkeypatch):
    explicit = tmp_path / "explicit"
    env_root = tmp_path / "env"
    monkeypatch.setenv("WEBBDUCK_PLUGINS_DIR", str(env_root))

    root, reason = installer._resolve_plugins_root(
        _args(plugins_dir=str(explicit), webbduck_dir=str(tmp_path / "webbduck"))
    )

    assert root == explicit.resolve()
    assert reason == "--plugins-dir"


def test_env_root_wins_over_webbduck_dir_to_match_runtime_discovery(tmp_path: Path, monkeypatch):
    env_root = tmp_path / "env-plugins"
    webbduck = tmp_path / "WebbDuck"
    monkeypatch.setenv("WEBBDUCK_PLUGINS_DIR", str(env_root))

    root, reason = installer._resolve_plugins_root(_args(webbduck_dir=str(webbduck)))

    assert root == env_root.resolve()
    assert "higher priority" in reason


def test_webbduck_local_root_is_used_without_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("WEBBDUCK_PLUGINS_DIR", raising=False)
    webbduck = tmp_path / "WebbDuck"

    root, reason = installer._resolve_plugins_root(_args(webbduck_dir=str(webbduck)))

    assert root == (webbduck / "plugins").resolve()
    assert reason == "--webbduck-dir"


def test_duplicate_plugin_roots_are_reported(tmp_path: Path, monkeypatch):
    env_root = tmp_path / "env"
    webbduck = tmp_path / "WebbDuck"
    home = tmp_path / "home"
    monkeypatch.setenv("WEBBDUCK_PLUGINS_DIR", str(env_root))
    monkeypatch.setattr(installer.Path, "home", classmethod(lambda cls: home))

    for root in ((webbduck / "plugins"), (home / ".webbduck" / "plugins")):
        plugin = root / "webapps" / "duckmotion"
        plugin.mkdir(parents=True)
        (plugin / "plugin.json").write_text("{}", encoding="utf-8")

    duplicates = installer._duplicate_plugin_roots(
        _args(webbduck_dir=str(webbduck)), env_root.resolve()
    )

    duplicate_roots = {root for root, _source in duplicates}
    assert (webbduck / "plugins").resolve() in duplicate_roots
    assert (home / ".webbduck" / "plugins").resolve() in duplicate_roots
