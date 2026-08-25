"""Validated execution-setting adapter for LTX-2.5 ConvRot companions.

DuckMotion never executes arbitrary Comfy workflow JSON. This module extracts a
small, allow-listed set of settings from either an explicit ``duckmotion_recipe``
manifest or the *active* graph of a structurally compatible exported workflow.

A critical rule here is that historical executor snapshots such as
``extra.prompt`` are not recipe evidence. Comfy workflow files can retain those
snapshots after the editable subgraph has changed. Mixing them with active nodes
can silently combine two different recipes (for example Euler from an old prompt
with Euler ancestral from the current subgraph).
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping


DEFAULT_STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
DEFAULT_STAGE2_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
DEFAULT_SAMPLER = "euler_ancestral"
DEFAULT_STAGE1_SAMPLER = DEFAULT_SAMPLER
DEFAULT_STAGE2_SAMPLER = DEFAULT_SAMPLER
DEFAULT_CFG = 1.0
DEFAULT_VIDEO_CFG = 1.0
DEFAULT_AUDIO_CFG = 1.0
DEFAULT_IMAGE_GUIDE_STRENGTH = 0.7
DEFAULT_UPSCALED_IMAGE_GUIDE_STRENGTH = 1.0
DEFAULT_NEGATIVE_PROMPT = "pc game, console game, video game, cartoon, childish, ugly"
DEFAULT_STAGE2_NOISE_POLICY = "fixed"
DEFAULT_STAGE2_FIXED_SEED = 42

_ALLOWED_SAMPLERS = {"euler", "euler_ancestral"}
_SAMPLER_ALIASES = {
    "euler": "euler",
    "euler a": "euler_ancestral",
    "euler_a": "euler_ancestral",
    "euler-a": "euler_ancestral",
    "euler ancestral": "euler_ancestral",
    "euler_ancestral": "euler_ancestral",
}
_ALLOWED_STAGE2_NOISE_POLICIES = {"increment", "same_seed", "fixed"}


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


def _node_type(node: Mapping[str, Any]) -> str:
    return str(node.get("type") or node.get("class_type") or "")


def _node_id(node: Mapping[str, Any]) -> str:
    return str(node.get("id") or node.get("node_id") or "")


def _node_enabled(node: Mapping[str, Any]) -> bool:
    # Comfy workflow mode 0 is active. Modes 2/4 are muted/bypassed and must not
    # participate in recipe inference. API-prompt nodes normally omit mode.
    mode = node.get("mode")
    return mode is None or mode == 0 or mode == "0"


def _graph_nodes(container: Any) -> list[dict[str, Any]]:
    if isinstance(container, Mapping):
        raw = container.get("nodes")
        if isinstance(raw, list):
            return [dict(node) for node in raw if isinstance(node, Mapping) and _node_enabled(node)]
        # API prompt format: {"12": {"class_type": ..., "inputs": ...}, ...}
        if raw is None:
            out: list[dict[str, Any]] = []
            for key, value in container.items():
                if isinstance(value, Mapping) and _node_type(value):
                    node = dict(value)
                    node.setdefault("id", key)
                    if _node_enabled(node):
                        out.append(node)
            return out
    return []


def _graph_score(nodes: list[dict[str, Any]]) -> int:
    types = {_node_type(node) for node in nodes}
    required = {"SamplerCustomAdvanced", "LTXVLatentUpsampler", "LTXVConcatAVLatent"}
    if not required.issubset(types):
        return -1
    score = len(required) * 10
    score += 6 if "LTXVImgToVideoInplace" in types else 0
    score += 5 if "LTXVDualCFGGuider" in types else 0
    score += 3 if "ManualSigmas" in types else 0
    return score


def _active_graph(config: Mapping[str, Any]) -> dict[str, Any]:
    """Select one authoritative editable graph without traversing stale metadata."""

    definitions = config.get("definitions")
    candidates: list[dict[str, Any]] = []
    if isinstance(definitions, Mapping):
        subgraphs = definitions.get("subgraphs")
        if isinstance(subgraphs, list):
            for subgraph in subgraphs:
                if not isinstance(subgraph, Mapping):
                    continue
                nodes = _graph_nodes(subgraph)
                score = _graph_score(nodes)
                if score >= 0:
                    candidates.append({"nodes": nodes, "links": subgraph.get("links") or [], "score": score})

    # A workflow with no compatible subgraph may expose the graph at top level.
    top_nodes = _graph_nodes(config)
    top_score = _graph_score(top_nodes)
    if top_score >= 0:
        candidates.append({"nodes": top_nodes, "links": config.get("links") or [], "score": top_score})

    if candidates:
        candidates.sort(key=lambda item: (item["score"], len(item["nodes"])), reverse=True)
        return candidates[0]

    # Keep a deliberately narrow fallback for small exported/test workflows.
    # Crucially this still never walks into ``extra`` / ``extra.prompt``.
    if top_nodes:
        return {"nodes": top_nodes, "links": config.get("links") or [], "score": 0}

    # API prompt may be nested under an explicit top-level prompt key. This is
    # supported only when the prompt itself is the node mapping, not extra.prompt.
    prompt = config.get("prompt")
    prompt_nodes = _graph_nodes(prompt) if isinstance(prompt, Mapping) else []
    if prompt_nodes:
        return {"nodes": prompt_nodes, "links": [], "score": 0}
    return {"nodes": [], "links": [], "score": -1}


def _nodes(graph: Mapping[str, Any], node_type: str) -> list[dict[str, Any]]:
    out = [node for node in graph.get("nodes", []) if _node_type(node) == node_type]
    return sorted(
        out,
        key=lambda node: (
            int(node.get("order", 1_000_000)) if str(node.get("order", "")).lstrip("-").isdigit() else 1_000_000,
            int(_node_id(node)) if _node_id(node).lstrip("-").isdigit() else 1_000_000,
        ),
    )


def _widgets(node: Mapping[str, Any]) -> list[Any]:
    values = node.get("widgets_values")
    return values if isinstance(values, list) else []


def _input_literal(node: Mapping[str, Any], name: str) -> Any:
    inputs = node.get("inputs")
    if isinstance(inputs, Mapping):
        value = inputs.get(name)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return None
        return value
    return None


def _link_origin_map(graph: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    links = graph.get("links")
    if not isinstance(links, list):
        return result
    for link in links:
        if isinstance(link, (list, tuple)) and len(link) >= 2:
            result[str(link[0])] = str(link[1])
        elif isinstance(link, Mapping):
            link_id = link.get("id")
            origin = link.get("origin_id") or link.get("origin_node")
            if link_id is not None and origin is not None:
                result[str(link_id)] = str(origin)
    return result


def _source_node(graph: Mapping[str, Any], node: Mapping[str, Any], input_name: str) -> dict[str, Any] | None:
    nodes_by_id = {_node_id(item): item for item in graph.get("nodes", []) if _node_id(item)}
    inputs = node.get("inputs")
    if isinstance(inputs, Mapping):
        value = inputs.get(input_name)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return nodes_by_id.get(str(value[0]))
        return None
    if not isinstance(inputs, list):
        return None
    link_id = None
    for item in inputs:
        if isinstance(item, Mapping) and str(item.get("name") or "") == input_name:
            link_id = item.get("link")
            break
    if link_id is None:
        return None
    origin_id = _link_origin_map(graph).get(str(link_id))
    return nodes_by_id.get(origin_id or "")


def _sampler_value(node: Mapping[str, Any]) -> str | None:
    literal = _input_literal(node, "sampler_name")
    if literal is not None:
        return normalize_sampler_name(literal)
    for value in _widgets(node):
        sampler = normalize_sampler_name(value)
        if sampler:
            return sampler
    return None


def _sigmas_value(node: Mapping[str, Any]) -> tuple[str, tuple[float, ...]] | None:
    literal = _input_literal(node, "sigmas")
    values = [literal] if literal is not None else _widgets(node)
    for value in values:
        parsed = _sigma_values(value)
        rendered = _sigma_string(value)
        if parsed is not None and rendered is not None:
            return rendered, parsed
    return None


def _dual_cfg_values(node: Mapping[str, Any]) -> tuple[float, float] | None:
    video = _as_float(_input_literal(node, "video_cfg"))
    audio = _as_float(_input_literal(node, "audio_cfg"))
    if video is not None and audio is not None:
        return video, audio
    values = [number for number in (_as_float(value) for value in _widgets(node)) if number is not None]
    if len(values) >= 2:
        return values[0], values[1]
    return None


def _single_cfg_value(node: Mapping[str, Any]) -> float | None:
    literal = _as_float(_input_literal(node, "cfg"))
    if literal is not None:
        return literal
    values = [number for number in (_as_float(value) for value in _widgets(node)) if number is not None]
    return values[0] if values else None


def _inplace_strength(node: Mapping[str, Any]) -> float | None:
    literal = _as_float(_input_literal(node, "strength"))
    if literal is not None:
        return literal if 0.0 <= literal <= 1.0 else None
    for value in _widgets(node):
        number = _as_float(value)
        if number is not None and 0.0 <= number <= 1.0:
            return number
    return None


def _sampling_stages(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve first/refinement samplers by the sigma schedule they consume."""

    stages: dict[str, dict[str, Any]] = {}
    for sampler_node in _nodes(graph, "SamplerCustomAdvanced"):
        sigma_node = _source_node(graph, sampler_node, "sigmas")
        sigma_info = _sigmas_value(sigma_node) if sigma_node else None
        if sigma_info is None:
            continue
        rendered, parsed = sigma_info
        stage = "stage1" if parsed[0] >= 0.95 else "stage2"
        if stage in stages:
            # Ambiguous active graph: refuse to combine two pipelines.
            return {}
        entry: dict[str, Any] = {"sampler_node": sampler_node, "sigmas": rendered}
        sampler_select = _source_node(graph, sampler_node, "sampler")
        if sampler_select:
            sampler = _sampler_value(sampler_select)
            if sampler:
                entry["sampler"] = sampler
        guider = _source_node(graph, sampler_node, "guider")
        if guider:
            entry["guider"] = guider
        noise = _source_node(graph, sampler_node, "noise")
        if noise:
            entry["noise"] = noise
        concat = _source_node(graph, sampler_node, "latent_image")
        if concat and _node_type(concat) == "LTXVConcatAVLatent":
            video_source = _source_node(graph, concat, "video_latent")
            if video_source:
                entry["video_source"] = video_source
        stages[stage] = entry
    return stages


def _workflow_sampling(graph: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stages = _sampling_stages(graph)
    if "stage1" in stages and "stage2" in stages:
        for stage in ("stage1", "stage2"):
            entry = stages[stage]
            result[f"{stage}_sigmas"] = entry["sigmas"]
            if entry.get("sampler"):
                result[f"{stage}_sampler"] = entry["sampler"]

        # Preserve the old common sampler field for metadata/backward-compatible
        # explicit manifests when both stages really do use one sampler.
        s1 = result.get("stage1_sampler")
        s2 = result.get("stage2_sampler")
        if s1 and s1 == s2:
            result["sampler"] = s1

        cfg_pairs: list[tuple[float, float]] = []
        for stage in ("stage1", "stage2"):
            guider = stages[stage].get("guider")
            if not guider:
                continue
            if _node_type(guider) == "LTXVDualCFGGuider":
                pair = _dual_cfg_values(guider)
                if pair:
                    cfg_pairs.append(pair)
            elif _node_type(guider) == "CFGGuider":
                cfg = _single_cfg_value(guider)
                if cfg is not None:
                    cfg_pairs.append((cfg, cfg))
        if cfg_pairs and all(pair == cfg_pairs[0] for pair in cfg_pairs):
            result["video_cfg"], result["audio_cfg"] = cfg_pairs[0]
            result["cfg"] = result["video_cfg"]

        # Resolve I2V strengths from the actual latent feeding each sampler,
        # rather than sorting nodes by execution order (which reverses the
        # official 0.7 -> 1.0 pair).
        for stage, target in (("stage1", "image_guide_strength"), ("stage2", "upscaled_image_guide_strength")):
            source = stages[stage].get("video_source")
            if source and _node_type(source) == "LTXVImgToVideoInplace":
                strength = _inplace_strength(source)
                if strength is not None:
                    result[target] = strength

        stage2_noise = stages["stage2"].get("noise")
        if stage2_noise and _node_type(stage2_noise) == "RandomNoise":
            seed = _as_int(_input_literal(stage2_noise, "noise_seed"))
            widgets = _widgets(stage2_noise)
            if seed is None and widgets:
                seed = _as_int(widgets[0])
            generation_mode = str(widgets[1]).strip().lower() if len(widgets) > 1 else ""
            if seed is not None and generation_mode in {"", "fixed"}:
                result["stage2_noise_policy"] = "fixed"
                result["stage2_fixed_seed"] = seed
        return result

    # Narrow fallback for older/simple workflows without link topology.
    sampler_values = {value for node in _nodes(graph, "KSamplerSelect") if (value := _sampler_value(node))}
    if len(sampler_values) == 1:
        sampler = next(iter(sampler_values))
        result.update({"sampler": sampler, "stage1_sampler": sampler, "stage2_sampler": sampler})

    schedules: list[tuple[str, tuple[float, ...]]] = []
    for node in _nodes(graph, "ManualSigmas"):
        info = _sigmas_value(node)
        if info and info not in schedules:
            schedules.append(info)
    high = [item for item in schedules if item[1][0] >= 0.95]
    low = [item for item in schedules if item[1][0] < 0.95]
    if len(high) == 1:
        result["stage1_sigmas"] = high[0][0]
    if len(low) == 1:
        result["stage2_sigmas"] = low[0][0]

    cfg_values = {
        cfg
        for node in _nodes(graph, "CFGGuider")
        if (cfg := _single_cfg_value(node)) is not None
    }
    if len(cfg_values) == 1:
        cfg = next(iter(cfg_values))
        result.update({"cfg": cfg, "video_cfg": cfg, "audio_cfg": cfg})

    strengths = [value for node in _nodes(graph, "LTXVImgToVideoInplace") if (value := _inplace_strength(node)) is not None]
    if strengths:
        result["image_guide_strength"] = strengths[0]
    if len(strengths) >= 2:
        result["upscaled_image_guide_strength"] = strengths[-1]

    noise_nodes = _nodes(graph, "RandomNoise")
    if len(noise_nodes) >= 2:
        seeds = []
        for node in noise_nodes[:2]:
            widgets = _widgets(node)
            seed = _as_int(_input_literal(node, "noise_seed"))
            if seed is None and widgets:
                seed = _as_int(widgets[0])
            seeds.append(seed)
        if None not in seeds:
            if seeds[0] == seeds[1]:
                result["stage2_noise_policy"] = "same_seed"
            elif seeds[1] == ((seeds[0] + 1) & ((1 << 64) - 1)):
                result["stage2_noise_policy"] = "increment"
    return result


def _workflow_negative_prompt(graph: Mapping[str, Any]) -> str | None:
    for conditioning in _nodes(graph, "LTXVConditioning"):
        negative_node = _source_node(graph, conditioning, "negative")
        if not negative_node or _node_type(negative_node) != "CLIPTextEncode":
            continue
        text = _input_literal(negative_node, "text")
        if text is None:
            widgets = _widgets(negative_node)
            text = widgets[0] if widgets else None
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


def _explicit_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    recipe = config.get("duckmotion_recipe")
    if not isinstance(recipe, Mapping):
        return {}

    result: dict[str, Any] = {}
    sampling = recipe.get("sampling")
    if isinstance(sampling, Mapping):
        common_sampler = normalize_sampler_name(sampling.get("sampler"))
        if common_sampler:
            result.update({"sampler": common_sampler, "stage1_sampler": common_sampler, "stage2_sampler": common_sampler})
        for key in ("stage1_sampler", "stage2_sampler"):
            sampler = normalize_sampler_name(sampling.get(key))
            if sampler:
                result[key] = sampler
        for key in ("stage1_sigmas", "stage2_sigmas"):
            sigmas = _sigma_string(sampling.get(key))
            if sigmas:
                result[key] = sigmas
        cfg = _as_float(sampling.get("cfg"))
        if cfg is not None and 0.0 <= cfg <= 20.0:
            result.update({"cfg": cfg, "video_cfg": cfg, "audio_cfg": cfg})
        for key in ("video_cfg", "audio_cfg"):
            value = _as_float(sampling.get(key))
            if value is not None and 0.0 <= value <= 100.0:
                result[key] = value
        noise_policy = str(sampling.get("stage2_noise_policy") or "").strip().lower()
        if noise_policy in _ALLOWED_STAGE2_NOISE_POLICIES:
            result["stage2_noise_policy"] = noise_policy
        fixed_seed = _as_int(sampling.get("stage2_fixed_seed"))
        if fixed_seed is not None:
            result["stage2_fixed_seed"] = fixed_seed & ((1 << 64) - 1)

    i2v = recipe.get("i2v")
    if isinstance(i2v, Mapping):
        for source_key, target_key in (
            ("stage1_guide_strength", "image_guide_strength"),
            ("stage2_guide_strength", "upscaled_image_guide_strength"),
        ):
            strength = _as_float(i2v.get(source_key))
            if strength is not None and 0.0 <= strength <= 1.0:
                result[target_key] = strength

    conditioning = recipe.get("conditioning")
    if isinstance(conditioning, Mapping):
        negative = conditioning.get("negative_prompt")
        if isinstance(negative, str):
            result["negative_prompt"] = negative.strip()
    return result


def extract_execution_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the safe runtime subset of a compatible companion workflow."""

    result: dict[str, Any] = {
        "sampler": DEFAULT_SAMPLER,
        "stage1_sampler": DEFAULT_STAGE1_SAMPLER,
        "stage2_sampler": DEFAULT_STAGE2_SAMPLER,
        "stage1_sigmas": DEFAULT_STAGE1_SIGMAS,
        "stage2_sigmas": DEFAULT_STAGE2_SIGMAS,
        "cfg": DEFAULT_CFG,
        "video_cfg": DEFAULT_VIDEO_CFG,
        "audio_cfg": DEFAULT_AUDIO_CFG,
        "image_guide_strength": DEFAULT_IMAGE_GUIDE_STRENGTH,
        "upscaled_image_guide_strength": DEFAULT_UPSCALED_IMAGE_GUIDE_STRENGTH,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "stage2_noise_policy": DEFAULT_STAGE2_NOISE_POLICY,
        "stage2_fixed_seed": DEFAULT_STAGE2_FIXED_SEED,
        "origin": "profile_default",
    }

    graph = _active_graph(config)
    inferred = _workflow_sampling(graph)
    negative = _workflow_negative_prompt(graph)
    if negative is not None:
        inferred["negative_prompt"] = negative
    if inferred:
        result.update(inferred)
        result["origin"] = "workflow_adapter"

    explicit = _explicit_recipe(config)
    if explicit:
        result.update(explicit)
        result["origin"] = "duckmotion_manifest"

    # Keep compatibility metadata internally consistent when stage samplers agree.
    if result.get("stage1_sampler") == result.get("stage2_sampler"):
        result["sampler"] = result["stage1_sampler"]
    return result
