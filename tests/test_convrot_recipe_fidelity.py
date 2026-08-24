from __future__ import annotations

from pathlib import Path

from ltx_convrot_quality import apply_quality_asset_policy
from ltx_convrot_recipe import extract_execution_recipe, normalize_sampler_name
from model_recipes import LTX25_CONVROT_TWO_STAGE_AV


def test_euler_a_alias_normalizes_to_comfy_sampler_name():
    assert normalize_sampler_name("Euler a") == "euler_ancestral"
    assert normalize_sampler_name("euler_ancestral") == "euler_ancestral"
    assert normalize_sampler_name("euler") == "euler"
    assert normalize_sampler_name("made-up") is None


def test_exported_workflow_custom_sampling_is_normalized():
    workflow = {
        "nodes": [
            {
                "id": 1,
                "order": 1,
                "type": "KSamplerSelect",
                "widgets_values": ["Euler a"],
            },
            {
                "id": 2,
                "order": 2,
                "type": "ManualSigmas",
                "widgets_values": ["1.0, 0.98, 0.8, 0.4, 0.0"],
            },
            {
                "id": 3,
                "order": 3,
                "type": "ManualSigmas",
                "widgets_values": ["0.82, 0.6, 0.2, 0.0"],
            },
            {
                "id": 4,
                "order": 4,
                "type": "CFGGuider",
                "widgets_values": [1.0],
            },
            {
                "id": 5,
                "order": 5,
                "type": "CFGGuider",
                "widgets_values": [1.0],
            },
            {
                "id": 6,
                "order": 6,
                "type": "LTXVImgToVideoInplace",
                "widgets_values": [0.95, False],
            },
            {
                "id": 7,
                "order": 7,
                "type": "LTXVImgToVideoInplace",
                "widgets_values": [1.0, False],
            },
            {
                "id": 8,
                "order": 8,
                "type": "RandomNoise",
                "widgets_values": [1234],
            },
            {
                "id": 9,
                "order": 9,
                "type": "RandomNoise",
                "widgets_values": [1234],
            },
        ]
    }

    recipe = extract_execution_recipe(workflow)
    assert recipe["origin"] == "workflow_adapter"
    assert recipe["sampler"] == "euler_ancestral"
    assert recipe["stage1_sigmas"] == "1.0, 0.98, 0.8, 0.4, 0.0"
    assert recipe["stage2_sigmas"] == "0.82, 0.6, 0.2, 0.0"
    assert recipe["cfg"] == 1.0
    assert recipe["image_guide_strength"] == 0.95
    assert recipe["upscaled_image_guide_strength"] == 1.0
    assert recipe["stage2_noise_policy"] == "same_seed"


def test_explicit_manifest_wins_over_workflow_inference():
    config = {
        "nodes": [
            {"type": "KSamplerSelect", "widgets_values": ["euler"]},
            {"type": "RandomNoise", "widgets_values": [10]},
            {"type": "RandomNoise", "widgets_values": [11]},
        ],
        "duckmotion_recipe": {
            "profile": LTX25_CONVROT_TWO_STAGE_AV.profile_id,
            "sampling": {
                "sampler": "Euler a",
                "stage1_sigmas": [1.0, 0.9, 0.0],
                "stage2_sigmas": [0.8, 0.0],
                "cfg": 1.25,
                "stage2_noise_policy": "same_seed",
            },
            "i2v": {
                "stage1_guide_strength": 1.0,
                "stage2_guide_strength": 1.0,
            },
        },
    }
    recipe = extract_execution_recipe(config)
    assert recipe["origin"] == "duckmotion_manifest"
    assert recipe["sampler"] == "euler_ancestral"
    assert recipe["cfg"] == 1.25
    assert recipe["stage2_noise_policy"] == "same_seed"
    assert recipe["stage1_sigmas"] == "1.0, 0.9, 0.0"
    assert recipe["stage2_sigmas"] == "0.8, 0.0"


def test_quality_policy_upgrades_only_known_legacy_standard_assets(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("DUCKMOTION_LTX_CONVROT_ASSET_POLICY", raising=False)
    checkpoint = tmp_path / "checkpoints" / "ltx" / "example-ltx25-convrot.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    preferred_vae = tmp_path / "support" / "ltx-2.5-video-vae-bf16.safetensors"
    preferred_upscaler = tmp_path / "support" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    preferred_vae.parent.mkdir(parents=True)
    preferred_vae.touch()
    preferred_upscaler.touch()

    state = {
        "ready": True,
        "config_path": str(checkpoint.with_suffix(".json")),
        "execution_profile": LTX25_CONVROT_TWO_STAGE_AV.profile_id,
        "asset_manifest": {
            "text_encoder": {"name": "custom-text.safetensors"},
            "latent_upscaler": {"name": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"},
            "video_vae": {"name": "ltx-2.5-video-vae-conv-bf16.safetensors"},
            "audio_vae": {"name": "custom-audio.safetensors"},
        },
        "asset_names": {
            "text_encoder": "custom-text.safetensors",
            "latent_upscaler": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "video_vae": "ltx-2.5-video-vae-conv-bf16.safetensors",
            "audio_vae": "custom-audio.safetensors",
        },
        "assets": {
            "text_encoder": "/models/custom-text.safetensors",
            "latent_upscaler": "/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "video_vae": "/models/ltx-2.5-video-vae-conv-bf16.safetensors",
            "audio_vae": "/models/custom-audio.safetensors",
        },
        "missing": [],
    }

    result = apply_quality_asset_policy(state, checkpoint, models_dir=str(tmp_path))
    assert result["asset_policy"] == "quality"
    assert Path(result["assets"]["video_vae"]).name == preferred_vae.name
    assert Path(result["assets"]["latent_upscaler"]).name == preferred_upscaler.name
    assert result["quality_upgrades"]["video_vae"]["from"] == "ltx-2.5-video-vae-conv-bf16.safetensors"
    assert result["quality_upgrades"]["latent_upscaler"]["from"] == "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    assert result["asset_names"]["text_encoder"] == "custom-text.safetensors"
    assert result["asset_names"]["audio_vae"] == "custom-audio.safetensors"


def test_recipe_asset_policy_can_preserve_legacy_assets(monkeypatch):
    monkeypatch.setenv("DUCKMOTION_LTX_CONVROT_ASSET_POLICY", "recipe")
    state = {
        "ready": True,
        "execution_profile": LTX25_CONVROT_TWO_STAGE_AV.profile_id,
        "asset_manifest": {
            "video_vae": {"name": "ltx-2.5-video-vae-conv-bf16.safetensors"},
        },
        "asset_names": {"video_vae": "ltx-2.5-video-vae-conv-bf16.safetensors"},
        "assets": {"video_vae": "/models/ltx-2.5-video-vae-conv-bf16.safetensors"},
        "missing": [],
    }
    result = apply_quality_asset_policy(state, "/tmp/missing.safetensors")
    assert result["asset_policy"] == "recipe"
    assert result["assets"]["video_vae"].endswith("ltx-2.5-video-vae-conv-bf16.safetensors")
