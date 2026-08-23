from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_execution_recipe_extension_contract_is_documented():
    source = (ROOT / "docs" / "EXECUTION_RECIPES.md").read_text(encoding="utf-8")
    assert "Users never select recipes" in source
    assert "If multiple profiles match" in source
    assert "Do not add a brand-name branch" in source
