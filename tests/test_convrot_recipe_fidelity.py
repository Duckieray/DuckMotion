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
            {"id": 1, "order": 1, "type": "KSamplerSelect", "widgets_values": ["Euler a"]},
            {"id": 2, "order": 2, "type": "ManualSigmas", "widgets_values": ["1.0, 0.98, 0.8, 0.4, 0.0"]},
            {"id": 3, "order": 3, "type": "ManualSigmas", "widgets_values": ["0.82, 0.6, 0.2, 0.0"]},
            {"id": 4, "order": 4, "type": "CFGGuider", "widgets_values": [1.0]},
            {"id": 5, "order": 5, "type": "CFGGuider", "widgets_values": [1.0]},
            {"id": 6, "order": 6, "type": "LTXVImgToVideoInplace", "widgets_values": [0.95, False]},
            {"id": 7, "order": 7, "type": "LTXVImgToVideoInplace", "widgets_values": [1.0, False]},
            {"id": 8, "order": 8, "type": "RandomNoise", "widgets_values": [1234]},
            {"id": 9, "order": 9, "type": "RandomNoise", "widgets_values": [1234]},
        ]
    }

    recipe = extract_execution_recipe(workflow)
    assert recipe["origin"] == "workflow_adapter"
    assert recipe["sampler"] == "euler_ancestral"
    assert recipe["stage1_sampler"] == "euler_ancestral"
    assert recipe["stage2_sampler"] == "euler_ancestral"
    assert recipe["stage1_sigmas"] == "1.0, 0.98, 0.8, 0.4, 0.0"
    assert recipe["stage2_sigmas"] == "0.82, 0.6, 0.2, 0.0"
    assert recipe["cfg"] == 1.0
    assert recipe["image_guide_strength"] == 0.95
    assert recipe["upscaled_image_guide_strength"] == 1.0
    assert recipe["stage2_noise_policy"] == "same_seed"


def _native_i2v_subgraph_fixture():
    nodes = [
        {"id": 1, "type": "EmptyLTXVLatentVideo", "mode": 0, "inputs": []},
        {
            "id": 2,
            "type": "LTXVImgToVideoInplace",
            "mode": 0,
            "order": 16,
            "inputs": [{"name": "latent", "link": 101}],
            "widgets_values": [0.7, False],
        },
        {
            "id": 3,
            "type": "LTXVConcatAVLatent",
            "mode": 0,
            "inputs": [{"name": "video_latent", "link": 102}],
        },
        {"id": 4, "type": "KSamplerSelect", "mode": 0, "widgets_values": ["euler_ancestral"]},
        {"id": 5, "type": "ManualSigmas", "mode": 0, "widgets_values": ["1.0, 0.99375, 0.725, 0.0"]},
        {"id": 6, "type": "LTXVDualCFGGuider", "mode": 0, "widgets_values": [1.0, 1.0]},
        {"id": 7, "type": "RandomNoise", "mode": 0, "widgets_values": [123, "randomize"]},
        {
            "id": 8,
            "type": "SamplerCustomAdvanced",
            "mode": 0,
            "inputs": [
                {"name": "noise", "link": 107},
                {"name": "guider", "link": 106},
                {"name": "sampler", "link": 104},
                {"name": "sigmas", "link": 105},
                {"name": "latent_image", "link": 103},
            ],
        },
        {"id": 9, "type": "LTXVSeparateAVLatent", "mode": 0, "inputs": []},
        {
            "id": 10,
            "type": "LTXVLatentUpsampler",
            "mode": 0,
            "inputs": [{"name": "samples", "link": 109}],
        },
        {
            "id": 11,
            "type": "LTXVImgToVideoInplace",
            "mode": 0,
            "order": 10,
            "inputs": [{"name": "latent", "link": 110}],
            "widgets_values": [1.0, False],
        },
        {
            "id": 12,
            "type": "LTXVConcatAVLatent",
            "mode": 0,
            "inputs": [{"name": "video_latent", "link": 111}],
        },
        {"id": 13, "type": "KSamplerSelect", "mode": 0, "widgets_values": ["euler_ancestral"]},
        {"id": 14, "type": "ManualSigmas", "mode": 0, "widgets_values": ["0.85, 0.7250, 0.4219, 0.0"]},
        {"id": 15, "type": "LTXVDualCFGGuider", "mode": 0, "widgets_values": [1.0, 1.0]},
        {"id": 16, "type": "RandomNoise", "mode": 0, "widgets_values": [42, "fixed"]},
        {
            "id": 17,
            "type": "SamplerCustomAdvanced",
            "mode": 0,
            "inputs": [
                {"name": "noise", "link": 117},
                {"name": "guider", "link": 116},
                {"name": "sampler", "link": 113},
                {"name": "sigmas", "link": 114},
                {"name": "latent_image", "link": 112},
            ],
        },
        {
            "id": 18,
            "type": "CLIPTextEncode",
            "mode": 0,
            "widgets_values": ["pc game, console game, video game, cartoon, childish, ugly"],
        },
        {
            "id": 19,
            "type": "LTXVConditioning",
            "mode": 0,
            "inputs": [{"name": "negative", "link": 118}],
        },
        {"id": 20, "type": "UNETLoader", "mode": 0},
        {"id": 21, "type": "CLIPLoader", "mode": 0},
        {"id": 22, "type": "VAELoader", "mode": 0},
        {"id": 23, "type": "LTXVEmptyLatentAudio", "mode": 0},
        {"id": 24, "type": "LatentUpscaleModelLoader", "mode": 0},
        {"id": 25, "type": "VAEDecodeTiled", "mode": 0},
        {"id": 26, "type": "LTXVAudioVAEDecode", "mode": 0},
    ]
    # Link shape: [link_id, origin_node_id, origin_slot, target_node_id, target_slot, type]
    links = [
        [101, 1, 0, 2, 0, "LATENT"],
        [102, 2, 0, 3, 0, "LATENT"],
        [103, 3, 0, 8, 4, "LATENT"],
        [104, 4, 0, 8, 2, "SAMPLER"],
        [105, 5, 0, 8, 3, "SIGMAS"],
        [106, 6, 0, 8, 1, "GUIDER"],
        [107, 7, 0, 8, 0, "NOISE"],
        [109, 9, 0, 10, 0, "LATENT"],
        [110, 10, 0, 11, 0, "LATENT"],
        [111, 11, 0, 12, 0, "LATENT"],
        [112, 12, 0, 17, 4, "LATENT"],
        [113, 13, 0, 17, 2, "SAMPLER"],
        [114, 14, 0, 17, 3, "SIGMAS"],
        [116, 15, 0, 17, 1, "GUIDER"],
        [117, 16, 0, 17, 0, "NOISE"],
        [118, 18, 0, 19, 1, "CONDITIONING"],
    ]
    return {"nodes": nodes, "links": links}


def test_active_subgraph_wins_over_stale_extra_prompt_and_preserves_stage_semantics():
    active = _native_i2v_subgraph_fixture()
    workflow = {
        "definitions": {"subgraphs": [active]},
        # This intentionally conflicts with the editable subgraph. Older adapter
        # code recursively scanned it and silently fell back to plain Euler.
        "extra": {
            "prompt": {
                "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
                "9": {"class_type": "LTXVImgToVideoInplace", "inputs": {"strength": 1.0}},
                "10": {"class_type": "LTXVImgToVideoInplace", "inputs": {"strength": 0.7}},
            }
        },
    }

    recipe = extract_execution_recipe(workflow)
    assert recipe["origin"] == "workflow_adapter"
    assert recipe["stage1_sampler"] == "euler_ancestral"
    assert recipe["stage2_sampler"] == "euler_ancestral"
    assert recipe["sampler"] == "euler_ancestral"
    assert recipe["image_guide_strength"] == 0.7
    assert recipe["upscaled_image_guide_strength"] == 1.0
    assert recipe["video_cfg"] == 1.0
    assert recipe["audio_cfg"] == 1.0
    assert recipe["negative_prompt"] == "pc game, console game, video game, cartoon, childish, ugly"
    assert recipe["stage2_noise_policy"] == "fixed"
    assert recipe["stage2_fixed_seed"] == 42


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
                "stage2_noise_policy": "fixed",
                "stage2_fixed_seed": 77,
            },
            "i2v": {
                "stage1_guide_strength": 0.8,
                "stage2_guide_strength": 1.0,
            },
            "conditioning": {"negative_prompt": "custom negative"},
        },
    }
    recipe = extract_execution_recipe(config)
    assert recipe["origin"] == "duckmotion_manifest"
    assert recipe["sampler"] == "euler_ancestral"
    assert recipe["stage1_sampler"] == "euler_ancestral"
    assert recipe["stage2_sampler"] == "euler_ancestral"
    assert recipe["cfg"] == 1.25
    assert recipe["video_cfg"] == 1.25
    assert recipe["audio_cfg"] == 1.25
    assert recipe["stage2_noise_policy"] == "fixed"
    assert recipe["stage2_fixed_seed"] == 77
    assert recipe["stage1_sigmas"] == "1.0, 0.9, 0.0"
    assert recipe["stage2_sigmas"] == "0.8, 0.0"
    assert recipe["image_guide_strength"] == 0.8
    assert recipe["negative_prompt"] == "custom negative"


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
