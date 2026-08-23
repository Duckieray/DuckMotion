from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import ltx_convrot_assets as assets


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "duckmotion_prepare_model_assets",
    ROOT / "tools" / "prepare_model_assets.py",
)
assert SPEC is not None and SPEC.loader is not None
prepare_assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_assets)


def _generic_recipe(checkpoint_name: str) -> dict:
    names = {
        "text_encoder": "gemma-ltx25-text-encoder.safetensors",
        "latent_upscaler": "generic-ltx-spatial-upscaler.safetensors",
        "video_vae": "generic-ltx25-video-vae.safetensors",
        "audio_vae": "generic-ltx25-audio-vae.safetensors",
    }
    nodes = [{"type": node_type} for node_type in sorted(assets.PROFILE_REQUIRED_NODES)]
    nodes.extend(
        [
            {
                "type": "CLIPLoader",
                "properties": {
                    "models": [
                        {
                            "name": names["text_encoder"],
                            "url": "https://huggingface.co/example-org/example-model/resolve/main/text_encoders/gemma-ltx25-text-encoder.safetensors",
                            "directory": "text_encoders",
                        }
                    ]
                },
                "widgets_values": [names["text_encoder"]],
            },
            {
                "type": "LatentUpscaleModelLoader",
                "properties": {
                    "models": [
                        {
                            "name": names["latent_upscaler"],
                            "url": "https://huggingface.co/example-org/example-model/resolve/main/generic-ltx-spatial-upscaler.safetensors",
                            "directory": "latent_upscale_models",
                        }
                    ]
                },
                "widgets_values": [names["latent_upscaler"]],
            },
            {"type": "VAELoader", "widgets_values": [names["video_vae"]]},
            {"type": "VAELoader", "widgets_values": [names["audio_vae"]]},
        ]
    )
    return {
        "checkpoint": checkpoint_name,
        "nodes": nodes,
        "asset_names": names,
    }


def test_generic_convrot_checkpoint_uses_recipe_profile_not_brand_name(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints" / "ltx"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "AcmeCinemaLTX25ConvRotQ8.safetensors"
    checkpoint.touch()

    recipe = _generic_recipe(checkpoint.name)
    (checkpoint_dir / "acme_recipe.json").write_text(json.dumps(recipe), encoding="utf-8")

    # Put the four support assets anywhere under the shared model root; no Comfy
    # category layout is required.
    for name in recipe["asset_names"].values():
        path = tmp_path / "arbitrary" / "support" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    result = assets.inspect_convrot_assets(checkpoint, models_dir=str(tmp_path))
    assert result["ready"] is True
    assert result["execution_profile"] == assets.SUPPORTED_EXECUTION_PROFILE
    assert result["missing"] == []
    assert all(result["assets"].values())


def test_recipe_declared_huggingface_urls_are_parsed_generically():
    assert prepare_assets._hf_source(
        "https://huggingface.co/example-org/example-model/resolve/main/path/to/asset.safetensors"
    ) == ("example-org/example-model", "path/to/asset.safetensors", "main")
    assert prepare_assets._hf_source("https://example.com/asset.safetensors") is None


def test_normal_setup_does_not_call_a_brand_specific_asset_preparer():
    setup_source = (ROOT / "tools" / "setup.py").read_text(encoding="utf-8")
    preparer_source = (ROOT / "tools" / "prepare_model_assets.py").read_text(encoding="utf-8")
    assert "prepare_model_assets.py" in setup_source
    assert "prepare_convrot_assets.py" not in setup_source
    assert "REDGraft" not in preparer_source


def test_core_asset_resolver_has_no_brand_specific_recipe_table():
    source = inspect.getsource(assets)
    assert "REDGRAFT_ASSET_SOURCES" not in source
    assert "is_redgraft_checkpoint" not in source
    assert "builtin:redgraft" not in source.lower()
