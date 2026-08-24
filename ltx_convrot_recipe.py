"""Validated execution-setting adapter for LTX-2.5 ConvRot companions.

DuckMotion never executes arbitrary Comfy workflow JSON.  This module extracts a
small, allow-listed set of settings from either an explicit ``duckmotion_recipe``
manifest or a structurally compatible exported workflow and normalizes them into
the runtime contract consumed by ``ltx_convrot_worker``.

The goal is to preserve checkpoint-author tuning (custom sigma schedules,
sampler, CFG, I2V guide strengths and stage-two noise semantics) without turning
a companion workflow into executable code.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping


DEFAULT_STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
DEFAULT_STAGE2_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
DEFAULT_SAMPLER = "euler"
DEFAULT_CFG = 1.0
DEFAULT_IMAGE_GUIDE_STRENGTH = 1.0
DEFAULT_UPSCALED_IMAGE_GUIDE_STRENGTH = 1.0
DEFAULT_STAGE2_NOISE_POLICY = "increment"

_ALLOWED_SAMPLERS = {"euler", "euler_ancestral"}
_SAMPLER_ALIASES = {
    "euler": "euler",
    "euler a": "euler_ancestral",
    "euler_a": "euler_ancestral",
    "euler-a": "euler_ancestral",
    "euler ancestral": "euler_ancestral",
    "euler_ancestral": "euler_ancestral",
}
_ALLOWED_STAGE2_NOISE_POLICIES = {"increment", "same_seed"}


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_dicts(child)


def _nodes(config: Mapping[str, Any], node_type: str) -> list[dict[str, Any]]:
    out = [
        node
        for node in _walk_dicts(config)
        if str(node.get("type") or node.get("class_type") or "") == node_type
    ]
    return sorted(
        out,
        key=lambda node: (
            int(node.get("order", 1_000_000)) if str(node.get("order", "")).lstrip("-").isdigit() else 1_000_000,
            int(node.get("id", 1_000_000)) if str(node.get("id", "")).lstrip("-").isdigit() else 1_000_000,
        ),
    )


def _widgets(node: Mapping[str, Any]) -> list[Any]:
    values = node.get("widgets_values")
    if isinstance(values, list):
        return values
    params = node.get("params")
    if isinstance(params, Mapping):
        return list(params.values())
    return []


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_sampler_name(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    normalized = _SAMPLER_ALIASES.get(raw)
    return normalized if normalized in _ALLOWED_SAMPLERS else None


def _sigma_values(value: Any) -> tuple[float, ...] | None:
    if isinstance(value, (list, tuple)):
        raw_values = list(value)
    elif isinstance(value, str):
        raw_values = [part for part in re.split(r"[\s,]+", value.strip()) if part]
    else:
        return None
    values: list[float] = []
    for raw in raw_values:
        number = _as_float(raw)
        if number is None:
            return None
        values.append(number)
    if len(values) < 2 or values[0] <= 0.0 or abs(values[-1]) > 1e-7:
        return None
    if any(values[index] < values[index + 1] for index in range(len(values) - 1)):
        return None
    return tuple(values)


def _sigma_string(value: Any) -> str | None:
    values = _sigma_values(value)
    if values is None:
        return None
    if isinstance(value, str):
        return value.strip()
    return ", ".join(str(item) for item in values)


def _explicit_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    recipe = config.get("duckmotion_recipe")
    if not isinstance(recipe, Mapping):
        return {}

    result: dict[str, Any] = {}
    sampling = recipe.get("sampling")
    if isinstance(sampling, Mapping):
        sampler = normalize_sampler_name(sampling.get("sampler"))
        if sampler:
            result["sampler"] = sampler
        for source_key, target_key in (
            ("stage1_sigmas", "stage1_sigmas"),
            ("stage2_sigmas", "stage2_sigmas"),
        ):
            sigmas = _sigma_string(sampling.get(source_key))
            if sigmas:
                result[target_key] = sigmas
        cfg = _as_float(sampling.get("cfg"))
        if cfg is not None and 0.0 <= cfg <= 20.0:
            result["cfg"] = cfg
        noise_policy = str(sampling.get("stage2_noise_policy") or "").strip().lower()
        if noise_policy in _ALLOWED_STAGE2_NOISE_POLICIES:
            result["stage2_noise_policy"] = noise_policy

    i2v = recipe.get("i2v")
    if isinstance(i2v, Mapping):
        for source_key, target_key in (
            ("stage1_guide_strength", "image_guide_strength"),
            ("stage2_guide_strength", "upscaled_image_guide_strength"),
        ):
            strength = _as_float(i2v.get(source_key))
            if strength is not None and 0.0 <= strength <= 1.0:
                result[target_key] = strength
    return result


def _workflow_sampler(config: Mapping[str, Any]) -> str | None:
    values: set[str] = set()
    for node in _nodes(config, "KSamplerSelect"):
        for value in _widgets(node):
            sampler = normalize_sampler_name(value)
            if sampler:
                values.add(sampler)
    return next(iter(values)) if len(values) == 1 else None


def _workflow_sigmas(config: Mapping[str, Any]) -> tuple[str | None, str | None]:
    schedules: list[tuple[str, tuple[float, ...]]] = []
    seen: set[tuple[float, ...]] = set()
    for node in _nodes(config, "ManualSigmas"):
        for value in _widgets(node):
            parsed = _sigma_values(value)
            rendered = _sigma_string(value)
            if parsed is None or rendered is None or parsed in seen:
                continue
            seen.add(parsed)
            schedules.append((rendered, parsed))
            break
    if len(schedules) < 2:
        return None, None

    # The first pass begins at (or very near) full noise and normally has more
    # transitions.  The refinement pass starts below full noise.  Refuse to
    # guess when the workflow does not expose one clear pair.
    high = [item for item in schedules if item[1][0] >= 0.95]
    low = [item for item in schedules if item[1][0] < 0.95]
    if len(high) == 1 and len(low) == 1:
        return high[0][0], low[0][0]
    if len(schedules) == 2:
        ordered = sorted(schedules, key=lambda item: (item[1][0], len(item[1])), reverse=True)
        if ordered[0][1][0] > ordered[1][1][0]:
            return ordered[0][0], ordered[1][0]
    return None, None


def _workflow_cfg(config: Mapping[str, Any]) -> float | None:
    values: set[float] = set()
    for node in _nodes(config, "CFGGuider"):
        for value in _widgets(node):
            number = _as_float(value)
            if number is not None and 0.0 <= number <= 20.0:
                values.add(round(number, 8))
    return next(iter(values)) if len(values) == 1 else None


def _workflow_guide_strengths(config: Mapping[str, Any]) -> tuple[float | None, float | None]:
    strengths: list[float] = []
    for node in _nodes(config, "LTXVImgToVideoInplace"):
        node_strength = None
        for value in _widgets(node):
            number = _as_float(value)
            if number is not None and 0.0 <= number <= 1.0:
                node_strength = number
                break
        if node_strength is not None:
            strengths.append(node_strength)
    if len(strengths) >= 2:
        return strengths[0], strengths[-1]
    if len(strengths) == 1:
        return strengths[0], None
    return None, None


def _workflow_stage2_noise_policy(config: Mapping[str, Any]) -> str | None:
    noise_nodes = _nodes(config, "RandomNoise")
    if len(noise_nodes) == 1:
        return "same_seed"
    if len(noise_nodes) < 2:
        return None
    seeds: list[int] = []
    for node in noise_nodes[:2]:
        seed = None
        for value in _widgets(node):
            seed = _as_int(value)
            if seed is not None:
                break
        if seed is None:
            return None
        seeds.append(seed)
    if seeds[1] == seeds[0]:
        return "same_seed"
    if seeds[1] == ((seeds[0] + 1) & ((1 << 64) - 1)):
        return "increment"
    return None


def extract_execution_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the safe runtime subset of a compatible companion workflow.

    Explicit ``duckmotion_recipe`` fields win over exported-workflow inference.
    Missing or ambiguous values fall back to the audited LTX-2.5 profile.
    """

    result: dict[str, Any] = {
        "sampler": DEFAULT_SAMPLER,
        "stage1_sigmas": DEFAULT_STAGE1_SIGMAS,
        "stage2_sigmas": DEFAULT_STAGE2_SIGMAS,
        "cfg": DEFAULT_CFG,
        "image_guide_strength": DEFAULT_IMAGE_GUIDE_STRENGTH,
        "upscaled_image_guide_strength": DEFAULT_UPSCALED_IMAGE_GUIDE_STRENGTH,
        "stage2_noise_policy": DEFAULT_STAGE2_NOISE_POLICY,
        "origin": "profile_default",
    }

    inferred: dict[str, Any] = {}
    sampler = _workflow_sampler(config)
    if sampler:
        inferred["sampler"] = sampler
    stage1_sigmas, stage2_sigmas = _workflow_sigmas(config)
    if stage1_sigmas:
        inferred["stage1_sigmas"] = stage1_sigmas
    if stage2_sigmas:
        inferred["stage2_sigmas"] = stage2_sigmas
    cfg = _workflow_cfg(config)
    if cfg is not None:
        inferred["cfg"] = cfg
    stage1_strength, stage2_strength = _workflow_guide_strengths(config)
    if stage1_strength is not None:
        inferred["image_guide_strength"] = stage1_strength
    if stage2_strength is not None:
        inferred["upscaled_image_guide_strength"] = stage2_strength
    noise_policy = _workflow_stage2_noise_policy(config)
    if noise_policy:
        inferred["stage2_noise_policy"] = noise_policy
    if inferred:
        result.update(inferred)
        result["origin"] = "workflow_adapter"

    explicit = _explicit_recipe(config)
    if explicit:
        result.update(explicit)
        result["origin"] = "duckmotion_manifest"
    return result
