"""Quality-oriented I2V reference-stability policy for LTX-2.5 ConvRot.

The execution recipe remains checkpoint/companion owned.  This module layers an
explicit user-facing stability choice on top of those values without making a
checkpoint brand part of runtime behavior.

``model`` preserves the companion's guide strengths/noise policy while using the
current LTXVAddGuide reference-conditioning path.
``identity`` strengthens the first-frame reference in both stages and reuses the
same stage-two seed to reduce high-resolution identity re-interpretation.
``locked`` adds the same source image as a final-frame reference as well.  It is
intended for tripod/locked-composition shots with small subject motion.
"""

from __future__ import annotations

from typing import Any, Mapping


MODEL_STABILITY = "model"
IDENTITY_STABILITY = "identity"
LOCKED_STABILITY = "locked"
DEFAULT_I2V_STABILITY = MODEL_STABILITY
I2V_STABILITY_MODES = (
    MODEL_STABILITY,
    IDENTITY_STABILITY,
    LOCKED_STABILITY,
)

_ALIASES = {
    "": DEFAULT_I2V_STABILITY,
    "default": MODEL_STABILITY,
    "model": MODEL_STABILITY,
    "model_default": MODEL_STABILITY,
    "identity": IDENTITY_STABILITY,
    "identity_stable": IDENTITY_STABILITY,
    "stable": IDENTITY_STABILITY,
    "locked": LOCKED_STABILITY,
    "locked_shot": LOCKED_STABILITY,
    "first_last": LOCKED_STABILITY,
}


def normalize_i2v_stability(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _ALIASES.get(raw, raw)
    return normalized if normalized in I2V_STABILITY_MODES else DEFAULT_I2V_STABILITY


def _bounded_strength(value: Any, fallback: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if 0.0 <= number <= 1.0 else fallback


def apply_i2v_stability(
    recipe: Mapping[str, Any],
    mode: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return effective recipe values plus a transparent override record."""

    normalized = normalize_i2v_stability(mode)
    effective = dict(recipe)
    overrides: dict[str, Any] = {}

    if normalized in {IDENTITY_STABILITY, LOCKED_STABILITY}:
        desired = {
            "image_guide_strength": 1.0,
            "upscaled_image_guide_strength": 1.0,
            "stage2_noise_policy": "same_seed",
        }
        for key, value in desired.items():
            if effective.get(key) != value:
                overrides[key] = {
                    "from": effective.get(key),
                    "to": value,
                }
            effective[key] = value

    effective["i2v_stability_mode"] = normalized
    return effective, overrides


def guide_plan(mode: Any, strength: Any) -> tuple[dict[str, Any], ...]:
    """Return the reference frames to add for one sampling stage.

    A locked shot uses the same source image as both first and last references.
    LTXVAddGuide keeps those references outside the generated timeline and marks
    their intended temporal positions through keyframe attention metadata.
    """

    normalized = normalize_i2v_stability(mode)
    first_strength = _bounded_strength(strength)
    guides: list[dict[str, Any]] = [
        {
            "role": "first",
            "frame_idx": 0,
            "strength": first_strength,
        }
    ]
    if normalized == LOCKED_STABILITY:
        guides.append(
            {
                "role": "last",
                "frame_idx": -1,
                "strength": 1.0,
            }
        )
    return tuple(guides)
