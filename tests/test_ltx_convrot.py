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
    _resolved_stage2_seed,
    _snap_dimension,
    _snap_frames,
    _stage2_seed,
)
from model_recipes import LTX25_CONVROT_TWO_STAGE_AV
from model_runtime import describe_video_model


def test_two_stage_av_profile_quality_defaults():
    assert HIGH_SIGMAS == "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
    assert LOW_SIGMAS == "0.85, 0.7250, 0.4219, 0.0"
    assert IMAGE_GUIDE_STRENGTH == 0.7
    assert UPSCALED_IMAGE_GUIDE_STRENGTH == 1.0
    assert IMAGE_PREPROCESS_LONG_EDGE == 1536
    assert IMAGE_PREPROCESS_COMPRESSION == 18
    assert LATENT_UPSCALE_METHOD == "ltx_learned_x2"
    assert LATENT_UPSCALE_SCALE == 2.0
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
    assert LTX25_CONVROT_TWO_STAGE_AV.constraints["i2v_stability_modes"] == ["model", "identity", "locked"]
    assert LTX25_CONVROT_TWO_STAGE_AV.constraints["sampling_schedule_locked"] is True
    assert LTX25_CONVROT_TWO_STAGE_AV.constraints["steps_locked"] is True
    assert LTX25_CONVROT_TWO_STAGE_AV.constraints["guidance_locked"] is False


def test_stage_two_seed_policy_supports_native_fixed_same_and_increment():
    assert _stage2_seed(0) == 1
    assert _stage2_seed(1234) == 1235
    assert _stage2_seed(UINT64_MASK) == 0
    assert _resolved_stage2_seed(1234, {"stage2_noise_policy": "same_seed"}) == 1234
    assert _resolved_stage2_seed(1234, {"stage2_noise_policy": "increment"}) == 1235
    assert _resolved_stage2_seed(1234, {"stage2_noise_policy": "fixed", "stage2_fixed_seed": 42}) == 42
    assert _effective_recipe({"execution_recipe": {"stage2_noise_policy": "fixed", "stage2_fixed_seed": 99}})["stage2_fixed_seed"] == 99


def test_user_guidance_overrides_both_native_dual_cfg_scales_without_unlocking_steps():
    recipe = _effective_recipe(
        {
            "execution_recipe": {
                "video_cfg": 1.0,
                "audio_cfg": 1.0,
                "origin": "workflow_adapter",
            },
            "guidance_scale": 1.35,
        }
    )
    assert recipe["cfg"] == 1.35
    assert recipe["video_cfg"] == 1.35
    assert recipe["audio_cfg"] == 1.35
    assert recipe["origin"] == "workflow_adapter+user_guidance"


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


def test_worker_source_mirrors_native_two_stage_i2v_and_keeps_locked_guides_optional():
    import ltx_convrot_worker

    source = inspect.getsource(ltx_convrot_worker._run)
    guide_source = inspect.getsource(ltx_convrot_worker._apply_guide_plan)
    inplace_source = inspect.getsource(ltx_convrot_worker._apply_inplace)
    guider_source = inspect.getsource(ltx_convrot_worker._dual_guider)
    module_source = inspect.getsource(ltx_convrot_worker)

    assert source.count('"CLIPTextEncode"') >= 2
    assert source.index('"CLIPTextEncode"') < source.index('"LTXVConditioning"')
    assert '"ConditioningZeroOut"' not in module_source
    assert '"VAELoader", vae_name=Path(assets["audio_vae"]).name' in source
    assert '"LTXVEmptyLatentAudio"' in source
    assert "audio_vae=audio_vae" in source
    assert '"LTXVImgToVideoInplace"' in inplace_source
    assert '"LTXVDualCFGGuider"' in guider_source
    assert '"CFGGuider",' not in module_source
    assert '"LTXVAddGuide"' in guide_source
    assert '"LTXVCropGuides"' in source  # advanced locked-shot path only
    assert '"LatentUpscaleBy"' not in module_source
    assert source.count('"LTXVLatentUpsampler"') == 1
    assert 'sampler_name=recipe["stage1_sampler"]' in source
    assert 'sampler_name=recipe["stage2_sampler"]' in source
    assert 'sigmas=recipe["stage1_sigmas"]' in source
    assert 'sigmas=recipe["stage2_sigmas"]' in source
    assert "noise_seed=stage2_seed" in source
    assert '"companion_execution_recipe": source_recipe' in source
    assert '"user_guidance_scale": request.get("guidance_scale")' in source
    assert '"reference_conditioning": reference_conditioning' in source
    assert '"sampling": "ltx25_convrot_two_stage_av_native"' in source
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
        "LTXVImgToVideoInplace",
        "LTXVDualCFGGuider",
        "LTXVAddGuide",
        "LTXVCropGuides",
        "LTXVLatentUpsampler",
        "VAEDecodeTiled",
        "LTXVAudioVAEDecode",
        "CreateVideo",
        "SaveVideo",
    ):
        assert node_name in LTX25_CONVROT_TWO_STAGE_AV.required_runtime_nodes
    assert "LatentUpscaleBy" not in LTX25_CONVROT_TWO_STAGE_AV.required_runtime_nodes
    assert "CFGGuider" not in LTX25_CONVROT_TWO_STAGE_AV.required_runtime_nodes


def test_backend_passes_guidance_to_isolated_worker_payload():
    source = inspect.getsource(LTX25ConvRotBackend.generate)
    assert '"guidance_scale": (' in source
    assert 'float(request["guidance_scale"])' in source


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
