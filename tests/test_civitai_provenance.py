from __future__ import annotations

import json
from pathlib import Path

import pytest

import civitai_provenance as civitai


def test_civitai_provider_requires_exact_sha256_and_caches_json_sidecars(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "anything.safetensors"
    checkpoint.write_bytes(b"weights")
    sha256 = "a" * 64
    payload = {
        "id": 222,
        "modelId": 111,
        "name": "Community release",
        "model": {"name": "Community Model"},
        "files": [
            {
                "id": 10,
                "name": "anything.safetensors",
                "hashes": {"SHA256": sha256.upper()},
                "downloadUrl": "https://civitai.com/api/download/models/222",
            },
            {
                "id": 11,
                "name": "workflow.json",
                "type": "Other",
                "sizeKB": 12,
                "hashes": {},
                # Deliberately stale/unpinned. The provider must canonicalize
                # this from the version/file IDs instead of trusting it.
                "downloadUrl": "https://civitai.com/api/download/models/222",
            },
            {
                "id": 12,
                "name": "preview.png",
                "type": "Other",
                "downloadUrl": "https://civitai.com/api/download/models/222",
            },
        ],
    }
    monkeypatch.setattr(civitai, "_get_json", lambda url, limit: payload)

    downloaded = []

    def fake_download(url: str, destination: Path) -> Path:
        downloaded.append((url, destination.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"nodes": []}), encoding="utf-8")
        return destination

    monkeypatch.setattr(civitai, "_download_recipe", fake_download)
    result = civitai.resolve_civitai_provenance(
        checkpoint,
        {"sha256": sha256},
        tmp_path / "cache",
    )

    assert result is not None
    assert result["source_id"] == "civitai:111@222"
    assert result["model_name"] == "Community Model"
    assert result["version_name"] == "Community release"
    assert len(result["recipe_paths"]) == 1
    assert Path(result["recipe_paths"][0]).name == "workflow.json"
    assert downloaded == [
        ("https://civitai.com/api/download/models/222?fileId=11", "workflow.json")
    ]
    assert Path(result["metadata_path"]).exists()


def test_civitai_file_download_url_falls_back_only_to_trusted_published_url():
    assert civitai._file_download_url(
        {"downloadUrl": "https://civitai.com/api/download/models/222?type=Other"},
        None,
    ) == "https://civitai.com/api/download/models/222?type=Other"
    assert civitai._file_download_url(
        {"downloadUrl": "https://example.com/workflow.json"},
        None,
    ) == ""


def test_civitai_provider_rejects_hash_response_without_exact_file_match(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setattr(
        civitai,
        "_get_json",
        lambda url, limit: {
            "id": 2,
            "modelId": 1,
            "files": [{"name": "other.safetensors", "hashes": {"SHA256": "b" * 64}}],
        },
    )
    with pytest.raises(ValueError, match="exact matching SHA256"):
        civitai.resolve_civitai_provenance(
            checkpoint,
            {"sha256": "a" * 64},
            tmp_path / "cache",
        )


def test_civitai_recipe_download_rejects_untrusted_source_before_network(tmp_path: Path):
    with pytest.raises(ValueError, match="untrusted"):
        civitai._download_recipe(
            "https://example.com/workflow.json",
            tmp_path / "workflow.json",
        )
