from __future__ import annotations

import json
from pathlib import Path

import ltx_convrot_assets as convrot
import model_provenance as provenance


def test_unbranded_convrot_recipe_can_resolve_entirely_from_cached_hash_provenance(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DUCKMOTION_PROVENANCE_CACHE", str(tmp_path / "provenance"))
    models = tmp_path / "models"
    checkpoint_dir = models / "checkpoints" / "ltx"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "AcmeCinemaLTX25ConvRotQ8.safetensors"
    checkpoint.write_bytes(b"fake checkpoint payload")

    names = {
        "text_encoder": "acme-gemma-ltx25.safetensors",
        "latent_upscaler": "acme-ltx-spatial-upscaler.safetensors",
        "video_vae": "acme-ltx-video-vae.safetensors",
        "audio_vae": "acme-ltx-audio-vae.safetensors",
    }
    for name in names.values():
        path = models / "support" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    remote_recipe = tmp_path / "provider-cache" / "recipe.json"
    remote_recipe.parent.mkdir(parents=True)
    remote_recipe.write_text(
        json.dumps(
            {
                "duckmotion_recipe": {
                    "profile": convrot.SUPPORTED_EXECUTION_PROFILE,
                    "assets": names,
                }
            }
        ),
        encoding="utf-8",
    )

    registry = provenance.CheckpointProvenanceRegistry()
    registry.register(
        provenance.CheckpointProvenanceProvider(
            provider_id="hash_catalog",
            resolve=lambda path, fingerprint, cache: {
                "source_id": "catalog:acme@1",
                "source_url": "https://catalog.example/acme/1",
                "model_name": "Acme Cinema",
                "version_name": "Q8 ConvRot",
                "recipe_paths": [str(remote_recipe)],
            },
        )
    )
    monkeypatch.setattr(provenance, "checkpoint_provenance_providers", registry)

    before = convrot.inspect_convrot_assets(checkpoint, models_dir=str(models))
    assert before["execution_profile"] is None
    assert before["recipe_origin"] is None

    resolved = provenance.prepare_checkpoint_provenance(checkpoint)
    assert resolved["matched"] is True

    after = convrot.inspect_convrot_assets(checkpoint, models_dir=str(models))
    assert after["ready"] is True
    assert after["execution_profile"] == convrot.SUPPORTED_EXECUTION_PROFILE
    assert after["recipe_origin"] == "provenance_cache"
    assert after["recipe_adapter"] == "duckmotion_manifest"
    assert after["provenance"]["provider"] == "hash_catalog"
    assert after["provenance"]["source_id"] == "catalog:acme@1"
    assert after["missing"] == []
