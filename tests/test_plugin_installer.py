from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "duckmotion_install_webbduck_plugin",
    ROOT / "tools" / "install_webbduck_plugin.py",
)
assert SPEC is not None and SPEC.loader is not None
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


def _make_source(root: Path) -> dict:
    manifest = {
        "id": "duckmotion",
        "entry": "index.html",
        "backend": "plugin_backend.py",
    }
    (root / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "plugin_backend.py").write_text("from helper import VALUE\n", encoding="utf-8")
    (root / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "README.md").write_text("DuckMotion\n", encoding="utf-8")
    (root / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (root / "ui").mkdir()
    (root / "ui" / "index.html").write_text("<html></html>\n", encoding="utf-8")
    (root / "runtime_requirements").mkdir()
    (root / "runtime_requirements" / "wan.txt").write_text("diffusers\n", encoding="utf-8")
    (root / "tools").mkdir()
    (root / "tools" / "development_only.py").write_text("raise SystemExit\n", encoding="utf-8")
    return manifest


def test_required_backend_comes_from_manifest_not_deleted_legacy_backend(tmp_path: Path):
    manifest = _make_source(tmp_path)
    required = installer._required_paths(tmp_path, manifest)
    assert tmp_path / "plugin_backend.py" in required
    assert tmp_path / "backend.py" not in required


def test_copy_plugin_tree_packages_modular_runtime_without_tooling(tmp_path: Path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    _make_source(source)

    installer._copy_plugin_tree(source, target)

    assert (target / "plugin.json").exists()
    assert (target / "plugin_backend.py").exists()
    assert (target / "helper.py").exists()
    assert (target / "ui" / "index.html").exists()
    assert (target / "runtime_requirements" / "wan.txt").exists()
    assert (target / "requirements.txt").exists()
    assert not (target / "backend.py").exists()
    assert not (target / "tools").exists()
