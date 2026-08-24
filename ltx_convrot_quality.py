"""Quality-first support-asset policy for LTX-2.5 ConvRot execution.

Recipe inspection remains authoritative for custom assets.  This layer upgrades
only known legacy standard assets that were historically used by DuckMotion's
compatibility profile (the lightweight Conv VAE and the older spatial upscaler)
to the current LTX-2.5 quality defaults.  Set
``DUCKMOTION_LTX_CONVROT_ASSET_POLICY=recipe`` as an advanced override to honor
those legacy declarations literally.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ltx_convrot_assets import candidate_search_roots, resolve_named_asset
from model_recipes import execution_profiles


LEGACY_QUALITY_ASSETS = {
    "video_vae": {
        "ltx-2.5-video-vae-conv-bf16.safetensors",
    },
    "latent_upscaler": {
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    },
}


def _policy() -> str:
    value = str(os.getenv("DUCKMOTION_LTX_CONVROT_ASSET_POLICY") or "quality").strip().lower()
    return value if value in {"quality", "recipe"} else "quality"


def apply_quality_asset_policy(
    state: dict[str, Any],
    checkpoint: str | Path,
    *,
    models_dir: str | None = None,
) -> dict[str, Any]:
    """Return an inspected asset state with quality-safe standard upgrades.

    Custom filenames are never replaced.  Only exact known legacy filenames are
    upgraded, and the profile remains the owner of the replacement filename/URL.
    """

    result = dict(state or {})
    result["asset_policy"] = _policy()
    if result["asset_policy"] != "quality":
        return result

    profile_id = str(result.get("execution_profile") or "").strip()
    profile = execution_profiles.get(profile_id)
    if profile is None:
        return result

    manifest = {
        key: dict(value or {})
        for key, value in dict(result.get("asset_manifest") or {}).items()
    }
    names = dict(result.get("asset_names") or {})
    resolved = dict(result.get("assets") or {})
    upgraded: dict[str, dict[str, str]] = {}
    roots = candidate_search_roots(checkpoint, models_dir=models_dir)

    for kind, legacy_names in LEGACY_QUALITY_ASSETS.items():
        declaration = manifest.get(kind, {})
        current_name = str(declaration.get("name") or names.get(kind) or "").strip()
        if not current_name or current_name.lower() not in legacy_names:
            continue
        preferred = dict(profile.asset_defaults.get(kind) or {})
        preferred_name = str(preferred.get("name") or "").strip()
        if not preferred_name or preferred_name.lower() == current_name.lower():
            continue

        manifest[kind] = {
            "name": preferred_name,
            "url": str(preferred.get("url") or "").strip(),
            "directory": str(preferred.get("directory") or "").strip(),
        }
        names[kind] = preferred_name
        preferred_path = resolve_named_asset(preferred_name, roots)
        resolved[kind] = str(preferred_path) if preferred_path else None
        upgraded[kind] = {"from": current_name, "to": preferred_name}

    if not upgraded:
        result["quality_upgrades"] = {}
        return result

    missing: list[str] = []
    for kind in profile.required_assets:
        if not resolved.get(kind):
            missing.append(str(names.get(kind) or f"<{kind} not declared in companion recipe>"))

    # Preserve recipe/checkpoint blockers from the underlying inspector while
    # replacing only asset-file missing entries with the effective quality set.
    original_missing = [str(item) for item in result.get("missing") or []]
    blockers = [item for item in original_missing if item.startswith("<")]
    missing = list(dict.fromkeys([*blockers, *missing]))

    checkpoint_path = Path(checkpoint).expanduser()
    recipe_ready = bool(
        checkpoint_path.exists()
        and checkpoint_path.is_file()
        and result.get("config_path")
        and result.get("execution_profile")
        and not blockers
    )
    result.update(
        {
            "asset_manifest": manifest,
            "asset_names": names,
            "assets": resolved,
            "missing": missing,
            "ready": recipe_ready and not missing,
            "quality_upgrades": upgraded,
        }
    )
    return result
