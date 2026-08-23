from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from ltx_convrot_assets import inspect_convrot_assets
from ltx_convrot_worker import (
    HIGH_SIGMAS,
    IMAGE_GUIDE_STRENGTH,
    IMAGE_PREPROCESS_COMPRESSION,
    IMAGE_PREPROCESS_LONG_EDGE,
    LATENT_UPSCALE_METHOD,
    LATENT_UPSCALE_SCALE,
    LOW_SIGMAS,
    UPSCALED_IMAGE_GUIDE_STRENGTH,
    VIDEO_DECODE_OVERLAP,
    VIDEO_DECODE_TEMPORAL_OVERLAP,
    VIDEO_DECODE_TEMPORAL_SIZE,
    VIDEO_DECODE_TILE_SIZE,
    _call_node,
    _snap_dimension,
    _snap_frames,
)
from model_runtime import describe_video_model


def test_redgraft_recipe_constants_are_locked():
    assert HIGH_SIGMAS == "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
    assert LOW_SIGMAS == "0.85, 0.7250, 0.4219, 0.0"
    assert IMAGE_GUIDE_STRENGTH == 0.7
    assert UPSCALED_IMAGE_GUIDE_STRENGTH == 1.0
    assert IMAGE_PREPROCESS_LONG_EDGE == 1536
    assert IMAGE_PREPROCESS_COMPRESSION == 18
    assert LATENT_UPSCALE_METHOD == "bicubic"
    assert LATENT_UPSCALE_SCALE == 0.5
    assert VIDEO_DECODE_TILE_SIZE == 480
    assert VIDEO_DECODE_OVERLAP == 96
    assert VIDEO_DECODE_TEMPORAL_SIZE == 96
    assert VIDEO_DECODE_TEMPORAL_OVERLAP == 24
    assert _snap_dimension(1152) == 1152
    assert _snap_dimension(768) == 768
    assert _snap_frames(10 * 24 + 1) == 241


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


def test_worker_source_keeps_redgraft_av_second_stage_and_decoder_order():
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
    assert "strength=UPSCALED_IMAGE_GUIDE_STRENGTH" in source
    assert source.index('"LTXVSeparateAVLatent", av_latent=stage2') < source.index('"VAEDecodeTiled"')
    assert "tile_size=VIDEO_DECODE_TILE_SIZE" in source
    assert "overlap=VIDEO_DECODE_OVERLAP" in source
    assert "temporal_size=VIDEO_DECODE_TEMPORAL_SIZE" in source
    assert "temporal_overlap=VIDEO_DECODE_TEMPORAL_OVERLAP" in source
    assert 'video_vae.decode(video_latent["samples"])' not in source


def test_companion_recipe_resolves_all_required_assets_from_standard_model_root(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints" / "ltx"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "REDGraft-ltx25-sulphur2-int8-convrot-ComfyMCP.safetensors"
    checkpoint.touch()

    names = {
        "text_encoder": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "latent_upscaler": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "video_vae": "ltx-2.5-video-vae-conv-bf16.safetensors",
        "audio_vae": "ltx-2.5-audio-vae-bf16.safetensors",
    }
    for folder, kind in (
        ("text_encoders", "text_encoder"),
        ("latent_upscale_models", "latent_upscaler"),
        ("vae", "video_vae"),
        ("vae", "audio_vae"),
    ):
        path = tmp_path / folder / names[kind]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    (checkpoint_dir / "redgraftLTX25Fast2K_ltx25RedgraftNSFW.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint.name,
                "text_encoder": names["text_encoder"],
                "latent_upscaler": names["latent_upscaler"],
                "video_vae": names["video_vae"],
                "audio_vae": names["audio_vae"],
            }
        ),
        encoding="utf-8",
    )

    # Readiness does not receive the persisted models_dir. The standard
    # models/checkpoints/... layout must therefore infer tmp_path as the shared
    # model root and resolve sibling text_encoders/vae/upscaler directories.
    result = inspect_convrot_assets(checkpoint)
    assert result["ready"] is True
    assert result["missing"] == []
    assert str(tmp_path) in result["search_roots"]
    for kind, name in names.items():
        assert Path(result["assets"][kind]).name == name


def test_convrot_descriptor_routes_privately_with_reference_defaults(tmp_path: Path):
    checkpoint = tmp_path / "REDGraft-ltx25-sulphur2-int8-convrot-ComfyMCP.safetensors"
    checkpoint.touch()
    descriptor = describe_video_model(str(checkpoint))

    assert descriptor.architecture == "ltx25"
    assert descriptor.backend == "ltx25_convrot"
    assert descriptor.detection["format"] == "int8_convrot"
    assert descriptor.capabilities.text_to_video is True
    assert descriptor.capabilities.image_to_video is True
    assert descriptor.capabilities.audio_output is True
    assert descriptor.defaults["width"] == 1152
    assert descriptor.defaults["height"] == 768
    assert descriptor.defaults["num_frames"] == 241
    assert descriptor.defaults["fps"] == 24
    assert descriptor.defaults["guidance_scale"] == 1.0
    public = descriptor.to_public_dict()
    assert "backend" not in public
    assert "architecture" not in public
