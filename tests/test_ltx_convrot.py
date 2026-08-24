from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from ltx_convrot_assets import inspect_convrot_assets
from ltx_convrot_backend import LTX25ConvRotBackend
from ltx_convrot_worker import (
    HIGH_SIGMAS,
    IMAGE_GUIDE_STRENGTH,
    IMAGE_PREPROCESS_COMPRESSION,
    IMAGE_PREPROCESS_LONG_EDGE,
    LATENT_UPSCALE_METHOD,
    LATENT_UPSCALE_SCALE,
    LOW_SIGMAS,
    UINT64_MASK,
    UPSCALED_IMAGE_GUIDE_STRENGTH,
    VIDEO_DECODE_OVERLAP,
    VIDEO_DECODE_TEMPORAL_OVERLAP,
    VIDEO_DECODE_TEMPORAL_SIZE,
    VIDEO_DECODE_TILE_SIZE,
    VIDEO_OUTPUT_CRF,
    _call_node,
    _effective_recipe,
    _snap_dimension,
    _snap_frames,
    _stage2_seed,
)
from model_recipes import LTX25_CONVROT_TWO_STAGE_AV
from model_runtime import describe_video_model


def test_two_stage_av_profile_quality_defaults():
    assert HIGH_SIGMAS == "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
    assert LOW_SIGMAS == "0.85, 0.7250, 0.4219, 0.0"
    assert IMAGE_GUIDE_STRENGTH == 1.0
    assert UPSCALED_IMAGE_GUIDE_STRENGTH == 1.0
    assert IMAGE_PREPROCESS_LONG_EDGE == 1536
    assert IMAGE_PREPROCESS_COMPRESSION == 18
    assert LATENT_UPSCALE_METHOD == "bicubic"
    assert LATENT_UPSCALE_SCALE == 0.5
    assert VIDEO_DECODE_TILE_SIZE == 480
    assert VIDEO_DECODE_OVERLAP == 96
    assert VIDEO_DECODE_TEMPORAL_SIZE == 96
    assert VIDEO_DECODE_TEMPORAL_OVERLAP == 24
    assert VIDEO_OUTPUT_CRF == 16
    assert _snap_dimension(1152) == 1152
    assert _snap_dimension(768) == 768
    assert _snap_frames(10 * 24 + 1) == 241
    assert LTX25_CONVROT_TWO_STAGE_AV.defaults["width"] == 1152
    assert LTX25_CONVROT_TWO_STAGE_AV.defaults["height"] == 768
    assert LTX25_CONVROT_TWO_STAGE_AV.defaults["num_frames"] == 241
    assert LTX25_CONVROT_TWO_STAGE_AV.defaults["num_inference_steps"] == 11
    assert LTX25_CONVROT_TWO_STAGE_AV.asset_defaults["video_vae"]["name"] == "ltx-2.5-video-vae-bf16.safetensors"
    assert LTX25_CONVROT_TWO_STAGE_AV.asset_defaults["latent_upscaler"]["name"] == "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"


def test_stage_two_seed_policy_can_preserve_or_increment_seed():
    assert _stage2_seed(0) == 1
    assert _stage2_seed(1234) == 1235
    assert _stage2_seed(UINT64_MASK) == 0
    assert _effective_recipe({"execution_recipe": {"stage2_noise_policy": "same_seed"}})["stage2_noise_policy"] == "same_seed"
    assert _effective_recipe({"execution_recipe": {"stage2_noise_policy": "increment"}})["stage2_noise_policy"] == "increment"


def test_call_node_supports_classic_and_modern_execute_contracts():
    class ClassicNode:
        FUNCTION = "run"

        def run(self, value):
            return (value + 1,)

    class ClassModernNode:
        @classmethod
        def execute(cls, value):
            return (value * 2,)

    class InstanceModernNode:
        def execute(self, value):
            return (value * 3,)

    nodes = SimpleNamespace(
        NODE_CLASS_MAPPINGS={
            "Classic": ClassicNode,
            "ClassModern": ClassModernNode,
            "InstanceModern": InstanceModernNode,
        }
    )

    assert _call_node(nodes, "Classic", value=2, ignored="filtered") == (3,)
    assert _call_node(nodes, "ClassModern", value=3, ignored="filtered") == (6,)
    assert _call_node(nodes, "InstanceModern", value=4, ignored="filtered") == (12,)


def test_worker_source_keeps_two_stage_av_order_and_uses_effective_recipe():
    import ltx_convrot_worker

    source = inspect.getsource(ltx_convrot_worker._run)
    assert source.index('"ConditioningZeroOut"') < source.index('"LTXVConditioning"')
    assert '"VAELoader", vae_name=Path(assets["audio_vae"]).name' in source
    assert '"LTXVEmptyLatentAudio"' in source
    assert "audio_vae=audio_vae" in source
    assert source.index('"LTXVCropGuides"') < source.index('"LTXVLatentUpsampler"')
    assert source.index('"LTXVLatentUpsampler"') < source.index('"LatentUpscaleBy"')
    assert "positive=stage2_positive" in source
    assert "negative=stage2_negative" in source
    assert 'strength=recipe["image_guide_strength"]' in source
    assert 'strength=recipe["upscaled_image_guide_strength"]' in source
    assert 'sampler_name=recipe["sampler"]' in source
    assert 'sigmas=recipe["stage1_sigmas"]' in source
    assert 'sigmas=recipe["stage2_sigmas"]' in source
    assert 'cfg=recipe["cfg"]' in source
    assert "noise_seed=stage2_seed" in source
    assert source.index('"LTXVSeparateAVLatent", av_latent=stage2') < source.index('"VAEDecodeTiled"')
    assert "tile_size=VIDEO_DECODE_TILE_SIZE" in source
    assert "overlap=VIDEO_DECODE_OVERLAP" in source
    assert "temporal_size=VIDEO_DECODE_TEMPORAL_SIZE" in source
    assert "temporal_overlap=VIDEO_DECODE_TEMPORAL_OVERLAP" in source
    assert 'video_vae.decode(video_latent["samples"])' not in source
    assert '"crf": VIDEO_OUTPUT_CRF' in source


def test_runtime_probe_uses_selected_profile_node_surface():
    source = inspect.getsource(LTX25ConvRotBackend._probe_runtime)
    assert "profile.required_runtime_nodes" in source
    assert "required_nodes.difference(nodes.NODE_CLASS_MAPPINGS)" in source
    assert "init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)" in source
    for node_name in (
        "LTXVEmptyLatentAudio",
        "LTXVConcatAVLatent",
        "LTXVCropGuides",
        "LTXVLatentUpsampler",
        "VAEDecodeTiled",
        "LTXVAudioVAEDecode",
        "CreateVideo",
        "SaveVideo",
    ):
        assert node_name in LTX25_CONVROT_TWO_STAGE_AV.required_runtime_nodes


def test_declarative_recipe_resolves_assets_without_brand_or_special_layout(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints" / "ltx"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "AcmeCinemaLTX25ConvRotQ8.safetensors"
    checkpoint.touch()

    names = {
        "text_encoder": "acme-text-encoder.safetensors",
        "latent_upscaler": "acme-spatial-upscaler.safetensors",
        "video_vae": "acme-video-vae.safetensors",
        "audio_vae": "acme-audio-vae.safetensors",
    }
    for name in names.values():
        path = tmp_path / "arbitrary" / "support" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    (checkpoint_dir / "acme.recipe.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint.name,
                "duckmotion_recipe": {
                    "profile": LTX25_CONVROT_TWO_STAGE_AV.profile_id,
                    "assets": names,
                },
            }
        ),
        encoding="utf-8",
    )

    result = inspect_convrot_assets(checkpoint, models_dir=str(tmp_path))
    assert result["ready"] is True
    assert result["execution_profile"] == LTX25_CONVROT_TWO_STAGE_AV.profile_id
    assert result["recipe_adapter"] == "duckmotion_manifest"
    assert result["missing"] == []
    for kind, name in names.items():
        assert Path(result["assets"][kind]).name == name


def test_convrot_descriptor_routes_format_without_choosing_recipe(tmp_path: Path):
    checkpoint = tmp_path / "AcmeCinemaLTX25ConvRotQ8.safetensors"
    checkpoint.touch()
    descriptor = describe_video_model(str(checkpoint))

    assert descriptor.architecture == "ltx25"
    assert descriptor.backend == "ltx25_convrot"
    assert descriptor.detection["format"] == "int8_convrot"
    assert descriptor.capabilities.text_to_video is True
    assert descriptor.capabilities.image_to_video is True
    assert descriptor.capabilities.audio_output is True
    assert descriptor.constraints["checkpoint_recipe_required"] is True
    public = descriptor.to_public_dict()
    assert "backend" not in public
    assert "architecture" not in public
    assert "execution_profile" not in public
