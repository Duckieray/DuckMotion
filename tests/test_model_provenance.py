from __future__ import annotations

from pathlib import Path

import model_provenance as provenance


def _registry(*providers):
    registry = provenance.CheckpointProvenanceRegistry()
    for provider in providers:
        registry.register(provider)
    return registry


def test_sha256_fingerprint_is_cached_until_file_changes(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DUCKMOTION_PROVENANCE_CACHE", str(tmp_path / "cache"))
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"first payload")

    first = provenance.checkpoint_fingerprint(checkpoint)
    second = provenance.checkpoint_fingerprint(checkpoint)
    assert first["cached"] is False
    assert second["cached"] is True
    assert first["sha256"] == second["sha256"]

    checkpoint.write_bytes(b"different payload")
    third = provenance.checkpoint_fingerprint(checkpoint)
    assert third["cached"] is False
    assert third["sha256"] != first["sha256"]


def test_exactly_one_provenance_provider_is_persisted_for_offline_use(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DUCKMOTION_PROVENANCE_CACHE", str(tmp_path / "cache"))
    checkpoint = tmp_path / "renamed-community-checkpoint.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    recipe = tmp_path / "cached-recipe.json"
    recipe.write_text("{}", encoding="utf-8")

    provider = provenance.CheckpointProvenanceProvider(
        provider_id="example_hash_catalog",
        resolve=lambda path, fingerprint, cache: {
            "source_id": "example:42",
            "source_url": "https://example.invalid/models/42",
            "model_name": "Unbranded Community Model",
            "version_name": "v1",
            "recipe_paths": [str(recipe)],
        },
    )
    monkeypatch.setattr(provenance, "checkpoint_provenance_providers", _registry(provider))

    result = provenance.prepare_checkpoint_provenance(checkpoint)
    assert result["matched"] is True
    assert result["cached"] is False
    cached = provenance.cached_provenance(checkpoint)
    assert cached is not None
    assert cached["provider"] == "example_hash_catalog"
    assert cached["source_id"] == "example:42"
    assert provenance.cached_recipe_paths(checkpoint) == (recipe,)


def test_materialized_recipe_provenance_skips_repeat_provider_lookup(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DUCKMOTION_PROVENANCE_CACHE", str(tmp_path / "cache"))
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}", encoding="utf-8")
    calls = []

    def resolve(path, fingerprint, cache):
        calls.append(fingerprint["sha256"])
        return {"source_id": "catalog:1", "recipe_paths": [str(recipe)]}

    provider = provenance.CheckpointProvenanceProvider(provider_id="catalog", resolve=resolve)
    monkeypatch.setattr(provenance, "checkpoint_provenance_providers", _registry(provider))

    first = provenance.prepare_checkpoint_provenance(checkpoint)
    second = provenance.prepare_checkpoint_provenance(checkpoint)
    assert first["cached"] is False
    assert second["cached"] is True
    assert len(calls) == 1


def test_ambiguous_provenance_is_not_cached(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DUCKMOTION_PROVENANCE_CACHE", str(tmp_path / "cache"))
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")

    providers = [
        provenance.CheckpointProvenanceProvider(
            provider_id=f"provider_{index}",
            resolve=lambda path, fingerprint, cache, index=index: {
                "source_id": f"source:{index}",
                "recipe_paths": [],
            },
        )
        for index in range(2)
    ]
    monkeypatch.setattr(provenance, "checkpoint_provenance_providers", _registry(*providers))

    result = provenance.prepare_checkpoint_provenance(checkpoint)
    assert result["matched"] is False
    assert result["ambiguous"] is True
    assert provenance.cached_provenance(checkpoint) is None


def test_changed_checkpoint_invalidates_cached_provenance(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DUCKMOTION_PROVENANCE_CACHE", str(tmp_path / "cache"))
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint-v1")
    provider = provenance.CheckpointProvenanceProvider(
        provider_id="catalog",
        resolve=lambda path, fingerprint, cache: {"source_id": "catalog:v1", "recipe_paths": []},
    )
    monkeypatch.setattr(provenance, "checkpoint_provenance_providers", _registry(provider))
    assert provenance.prepare_checkpoint_provenance(checkpoint)["matched"] is True
    assert provenance.cached_provenance(checkpoint) is not None

    checkpoint.write_bytes(b"checkpoint-v2")
    provenance.checkpoint_fingerprint(checkpoint)
    assert provenance.cached_provenance(checkpoint) is None
