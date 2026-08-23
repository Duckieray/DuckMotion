from __future__ import annotations

import importlib.util
from pathlib import Path

import ltx_convrot_assets as convrot_assets
from model_recipes import LTX25_CONVROT_TWO_STAGE_AV


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "duckmotion_prepare_model_assets_profile_defaults",
    ROOT / "tools" / "prepare_model_assets.py",
)
assert SPEC is not None and SPEC.loader is not None
prepare_assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_assets)


def test_convrot_profile_owns_standard_support_sources_not_checkpoint_brand():
    profile = LTX25_CONVROT_TWO_STAGE_AV
    assert set(profile.asset_defaults) == set(profile.required_assets)
    joined = repr(profile.asset_defaults).lower()
    assert "redgraft" not in joined
    assert "lightricks/ltx-2.5" in joined
    assert "lightricks/ltx-2.3" in joined


def test_profile_source_fills_missing_url_for_same_standard_asset():
    fallback = LTX25_CONVROT_TWO_STAGE_AV.asset_defaults["video_vae"]
    name, url = prepare_assets._asset_declaration(
        LTX25_CONVROT_TWO_STAGE_AV.profile_id,
        "video_vae",
        {"name": fallback["name"], "url": ""},
    )
    assert name == fallback["name"]
    assert url == fallback["url"]


def test_profile_source_never_substitutes_a_custom_recipe_asset():
    name, url = prepare_assets._asset_declaration(
        LTX25_CONVROT_TWO_STAGE_AV.profile_id,
        "video_vae",
        {"name": "community-special-video-vae.safetensors", "url": ""},
    )
    assert name == "community-special-video-vae.safetensors"
    assert url == ""


def test_sparse_recipe_manifest_is_normalized_with_profile_defaults_for_readiness():
    profile = LTX25_CONVROT_TWO_STAGE_AV
    config = {
        "duckmotion_recipe": {
            "profile": profile.profile_id,
            "assets": {
                "text_encoder": {
                    "name": profile.asset_defaults["text_encoder"]["name"],
                }
            },
        }
    }
    manifest = convrot_assets.extract_asset_manifest(config, "UnbrandedConvRot.safetensors")
    assert set(manifest) == set(profile.required_assets)
    for kind in profile.required_assets:
        assert manifest[kind]["name"] == profile.asset_defaults[kind]["name"]
        assert manifest[kind]["url"] == profile.asset_defaults[kind]["url"]


def test_manifest_normalization_preserves_custom_asset_identity():
    profile = LTX25_CONVROT_TWO_STAGE_AV
    config = {
        "duckmotion_recipe": {
            "profile": profile.profile_id,
            "assets": {
                "video_vae": {"name": "community-special-video-vae.safetensors"},
            },
        }
    }
    manifest = convrot_assets.extract_asset_manifest(config, "UnbrandedConvRot.safetensors")
    assert manifest["video_vae"]["name"] == "community-special-video-vae.safetensors"
    assert manifest["video_vae"]["url"] == ""
