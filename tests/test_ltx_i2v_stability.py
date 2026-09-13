from __future__ import annotations

from ltx_i2v_stability import (
    DEFAULT_I2V_STABILITY,
    IDENTITY_STABILITY,
    LOCKED_STABILITY,
    MODEL_STABILITY,
    apply_i2v_stability,
    guide_plan,
    normalize_i2v_stability,
)


def test_stability_mode_aliases_are_normalized_without_model_branding():
    assert normalize_i2v_stability(None) == DEFAULT_I2V_STABILITY
    assert normalize_i2v_stability("model default") == MODEL_STABILITY
    assert normalize_i2v_stability("identity stable") == IDENTITY_STABILITY
    assert normalize_i2v_stability("locked shot") == LOCKED_STABILITY
    assert normalize_i2v_stability("unknown") == MODEL_STABILITY


def test_model_mode_preserves_companion_recipe_values():
    recipe = {
        "image_guide_strength": 0.8,
        "upscaled_image_guide_strength": 0.7,
        "stage2_noise_policy": "increment",
        "sampler": "euler",
    }
    effective, overrides = apply_i2v_stability(recipe, "model")

    assert effective["image_guide_strength"] == 0.8
    assert effective["upscaled_image_guide_strength"] == 0.7
    assert effective["stage2_noise_policy"] == "increment"
    assert effective["sampler"] == "euler"
    assert effective["i2v_stability_mode"] == "model"
    assert overrides == {}


def test_identity_mode_strengthens_both_stages_and_reuses_noise_seed():
    recipe = {
        "image_guide_strength": 1.0,
        "upscaled_image_guide_strength": 0.7,
        "stage2_noise_policy": "increment",
    }
    effective, overrides = apply_i2v_stability(recipe, "identity")

    assert effective["image_guide_strength"] == 1.0
    assert effective["upscaled_image_guide_strength"] == 1.0
    assert effective["stage2_noise_policy"] == "same_seed"
    assert overrides == {
        "upscaled_image_guide_strength": {"from": 0.7, "to": 1.0},
        "stage2_noise_policy": {"from": "increment", "to": "same_seed"},
    }


def test_locked_mode_adds_same_source_at_first_and_last_frame():
    guides = guide_plan("locked", 0.7)

    assert guides == (
        {"role": "first", "frame_idx": 0, "strength": 0.7},
        {"role": "last", "frame_idx": -1, "strength": 1.0},
    )


def test_non_locked_modes_use_only_first_frame_reference():
    assert guide_plan("model", 0.7) == (
        {"role": "first", "frame_idx": 0, "strength": 0.7},
    )
    assert guide_plan("identity", 1.0) == (
        {"role": "first", "frame_idx": 0, "strength": 1.0},
    )
