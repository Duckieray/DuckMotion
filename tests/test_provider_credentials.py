from __future__ import annotations

import json
import os
from pathlib import Path

import provider_credentials as credentials


def _write_settings(path: Path, *, hf: str = "", civitai: str = "") -> None:
    providers = {}
    if hf:
        providers["huggingface"] = {"token": hf}
    if civitai:
        providers["civitai"] = {"token": civitai}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "providers": providers}), encoding="utf-8")


def _clear_provider_env(monkeypatch) -> None:
    for name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "CIVITAI_TOKEN",
        "CIVITAI_API_TOKEN",
        "CIVITAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_duckmotion_reads_webbduck_settings_without_server_or_shell_exports(monkeypatch, tmp_path: Path):
    _clear_provider_env(monkeypatch)
    path = tmp_path / "provider_credentials.json"
    monkeypatch.setenv("WEBBDUCK_CREDENTIALS_FILE", str(path))
    _write_settings(path, hf="hf-from-settings", civitai="civitai-from-settings")

    assert credentials.provider_token("huggingface") == "hf-from-settings"
    assert credentials.provider_token("civitai") == "civitai-from-settings"

    credentials.apply_provider_credentials_to_environment()
    assert os.environ["HF_TOKEN"] == "hf-from-settings"
    assert os.environ["CIVITAI_TOKEN"] == "civitai-from-settings"


def test_environment_aliases_remain_advanced_overrides(monkeypatch, tmp_path: Path):
    _clear_provider_env(monkeypatch)
    path = tmp_path / "provider_credentials.json"
    monkeypatch.setenv("WEBBDUCK_CREDENTIALS_FILE", str(path))
    _write_settings(path, hf="hf-settings", civitai="civitai-settings")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf-env")
    monkeypatch.setenv("CIVITAI_API_KEY", "civitai-env")

    assert credentials.provider_token("huggingface") == "hf-env"
    assert credentials.provider_token("civitai") == "civitai-env"
    credentials.apply_provider_credentials_to_environment()
    assert "HF_TOKEN" not in os.environ
    assert "CIVITAI_TOKEN" not in os.environ


def test_missing_or_invalid_settings_file_means_anonymous_access(monkeypatch, tmp_path: Path):
    _clear_provider_env(monkeypatch)
    path = tmp_path / "missing.json"
    monkeypatch.setenv("WEBBDUCK_CREDENTIALS_FILE", str(path))
    assert credentials.provider_token("huggingface") == ""
    assert credentials.provider_token("civitai") == ""

    path.write_text("not-json", encoding="utf-8")
    assert credentials.provider_token("huggingface") == ""
    assert credentials.provider_token("civitai") == ""
