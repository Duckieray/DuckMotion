#!/usr/bin/env python3
"""Human-friendly DuckMotion installation/model readiness report.

This command does not load model weights, access provenance networks, or submit
generation jobs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_paths import RUNTIME_ENV_VARS, configure_default_runtime_env, runtime_status

configure_default_runtime_env()

from ltx_backend import ensure_registered as ensure_ltx_registered
from ltx_convrot_backend import ensure_registered as ensure_ltx_convrot_registered
from model_discovery import discover_video_models
from model_runtime import backend_resolver, describe_video_model
from wan_backend import ensure_registered as ensure_wan_registered


CONFIG_FILE = Path.home() / ".webbduck" / "plugin_state" / "duckmotion_config.json"


def _load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {"model_id_or_path": "", "models_dir": "", "output_dir": ""}
    try:
        value = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"model_id_or_path": "", "models_dir": "", "output_dir": ""}
    if not isinstance(value, dict):
        value = {}
    return {
        "model_id_or_path": str(value.get("model_id_or_path") or "").strip(),
        "models_dir": str(value.get("models_dir") or "").strip(),
        "output_dir": str(value.get("output_dir") or "").strip(),
    }


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def _short_reason(payload: dict) -> str:
    reason = str(payload.get("reason") or "").strip()
    if reason:
        return reason
    return "ready" if payload.get("ready") else "not ready"


def _print_convrot_details(readiness: dict, assets: dict) -> None:
    recipe = readiness.get("execution_recipe")
    if isinstance(recipe, dict):
        origin = str(recipe.get("origin") or "profile_default")
        sampler1 = str(recipe.get("stage1_sampler") or recipe.get("sampler") or "")
        sampler2 = str(recipe.get("stage2_sampler") or recipe.get("sampler") or "")
        video_cfg = recipe.get("video_cfg", recipe.get("cfg"))
        audio_cfg = recipe.get("audio_cfg", recipe.get("cfg"))
        noise = str(recipe.get("stage2_noise_policy") or "")
        if noise == "fixed" and recipe.get("stage2_fixed_seed") is not None:
            noise = f"fixed:{recipe['stage2_fixed_seed']}"
        guide1 = recipe.get("image_guide_strength")
        guide2 = recipe.get("upscaled_image_guide_strength")
        print(
            "      effective tuning: "
            f"{origin}; samplers={sampler1}/{sampler2}; "
            f"cfg(video/audio)={video_cfg}/{audio_cfg}; "
            f"i2v={guide1}/{guide2}; stage2_noise={noise}"
        )
        if recipe.get("negative_prompt"):
            print(f"      negative conditioning: {recipe['negative_prompt']}")
        if recipe.get("stage1_sigmas"):
            print(f"      stage1 sigmas: {recipe['stage1_sigmas']}")
        if recipe.get("stage2_sigmas"):
            print(f"      stage2 sigmas: {recipe['stage2_sigmas']}")

    policy = str(assets.get("asset_policy") or "").strip()
    names = assets.get("asset_names") if isinstance(assets.get("asset_names"), dict) else {}
    if policy:
        print(f"      asset policy: {policy}")
    if names.get("video_vae"):
        print(f"      video VAE: {names['video_vae']}")
    if names.get("latent_upscaler"):
        print(f"      latent upscaler: {names['latent_upscaler']}")
    upgrades = assets.get("quality_upgrades") if isinstance(assets.get("quality_upgrades"), dict) else {}
    for kind, change in upgrades.items():
        if not isinstance(change, dict):
            continue
        print(
            f"      quality upgrade ({kind}): "
            f"{change.get('from')} -> {change.get('to')}"
        )


def main() -> int:
    config = _load_config()
    models_dir = str(config.get("models_dir") or "").strip()

    print("DuckMotion doctor")
    print("=================")
    print(f"Config: {CONFIG_FILE}")
    if models_dir:
        path = Path(models_dir).expanduser()
        print(f"Models: {_mark(path.exists() and path.is_dir())} {path}")
    else:
        print("Models: ✗ not configured (run: python tools/setup.py --models /path/to/models)")

    print("")
    print("Runtimes")
    runtime_ok = True
    for runtime in RUNTIME_ENV_VARS:
        status = runtime_status(runtime)
        exists = bool(status["exists"])
        runtime_ok = runtime_ok and exists
        print(
            f"  {_mark(exists)} {runtime:<13} {status['python']}"
            + ("" if status["source"] == "default" else " (override)")
        )

    ensure_wan_registered()
    ensure_ltx_registered()
    ensure_ltx_convrot_registered()

    catalog = discover_video_models(config)
    items = [item for item in catalog.get("items") or [] if isinstance(item, dict)]
    print("")
    print(f"Models discovered: {len(items)}")
    print(f"Hugging Face cache: {catalog.get('hf_cache')}")

    ready_count = 0
    for item in items:
        source = str(item.get("source") or item.get("repo_id") or item.get("name") or "").strip()
        name = str(item.get("name") or Path(source).name or source).strip()
        descriptor = describe_video_model(source, name=name)
        if not descriptor.supported:
            print(f"  ! {name}: detected, runtime not implemented")
            continue
        try:
            readiness = backend_resolver.readiness(descriptor)
        except Exception as exc:
            readiness = {"ready": False, "reason": str(exc or exc.__class__.__name__)}
        ready = bool(readiness.get("ready"))
        if ready:
            ready_count += 1
        print(f"  {_mark(ready)} {name}: {_short_reason(readiness)}")

        source_format = str(readiness.get("source_format") or "").strip()
        profile = str(readiness.get("execution_profile") or "").strip()
        assets = readiness.get("assets") if isinstance(readiness.get("assets"), dict) else None
        if source_format:
            print(f"      format: {source_format}")
        if profile:
            print(f"      recipe: {profile}")
        elif source_format and not ready:
            if assets and assets.get("config_path"):
                print("      recipe: unresolved/unsupported")
            else:
                print("      recipe: not resolved")

        if assets:
            provenance = assets.get("provenance") if isinstance(assets.get("provenance"), dict) else None
            if provenance:
                provider = str(provenance.get("provider") or "cached")
                source_id = str(provenance.get("source_id") or provenance.get("source_url") or "matched")
                print(f"      provenance: {provider} ({source_id})")
            recipe_origin = str(assets.get("recipe_origin") or "").strip()
            if recipe_origin:
                print(f"      recipe source: {recipe_origin}")
            if source_format == "int8_convrot":
                _print_convrot_details(readiness, assets)
            if not assets.get("ready"):
                for missing in assets.get("missing") or []:
                    print(f"      missing: {missing}")

    print("")
    if not items:
        print("No DuckMotion video models were discovered.")
    elif ready_count:
        print(f"Ready models: {ready_count}/{len(items)}")
    else:
        print("No discovered model is generation-ready yet.")

    if not runtime_ok:
        print("Run: python tools/setup.py --models /path/to/models")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
