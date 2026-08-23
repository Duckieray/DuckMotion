"""Architecture-neutral execution recipe contracts for DuckMotion.

A model descriptor answers *what the checkpoint is* (architecture/format and
capabilities). An execution recipe answers *how a supported runtime should run
it*. Product/display names never participate in recipe selection.

Recipes are declarative contracts. A companion file may name a profile directly,
or an adapter may infer a profile from trusted structural evidence such as the
set of node types in an exported workflow. DuckMotion never executes arbitrary
recipe/workflow code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ExecutionProfile:
    profile_id: str
    architecture: str
    source_format: str
    required_assets: tuple[str, ...] = ()
    evidence_node_types: frozenset[str] = frozenset()
    required_runtime_nodes: frozenset[str] = frozenset()


class ExecutionProfileRegistry:
    def __init__(self) -> None:
        self._profiles: dict[str, ExecutionProfile] = {}

    def register(self, profile: ExecutionProfile) -> None:
        profile_id = str(profile.profile_id or "").strip()
        if not profile_id:
            raise ValueError("Execution profile must define profile_id")
        if profile_id in self._profiles:
            raise ValueError(f"Duplicate DuckMotion execution profile: {profile_id}")
        self._profiles[profile_id] = profile

    def get(self, profile_id: str | None) -> ExecutionProfile | None:
        return self._profiles.get(str(profile_id or "").strip())

    def compatible(
        self,
        *,
        architecture: str,
        source_format: str,
    ) -> tuple[ExecutionProfile, ...]:
        architecture = str(architecture or "").lower()
        source_format = str(source_format or "").lower()
        return tuple(
            profile
            for profile in self._profiles.values()
            if profile.architecture.lower() == architecture
            and profile.source_format.lower() == source_format
        )

    def infer_from_node_types(
        self,
        *,
        architecture: str,
        source_format: str,
        node_types: Iterable[str],
    ) -> ExecutionProfile | None:
        evidence = {str(value) for value in node_types if str(value)}
        matches = [
            profile
            for profile in self.compatible(
                architecture=architecture,
                source_format=source_format,
            )
            if profile.evidence_node_types
            and profile.evidence_node_types.issubset(evidence)
        ]
        # Never guess when two installed profiles match the same evidence.
        return matches[0] if len(matches) == 1 else None

    def ids(self) -> tuple[str, ...]:
        return tuple(self._profiles.keys())


execution_profiles = ExecutionProfileRegistry()


_LTX25_CONVROT_EVIDENCE_NODES = frozenset(
    {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "ConditioningZeroOut",
        "LTXVConditioning",
        "LTXVEmptyLatentAudio",
        "LTXVConcatAVLatent",
        "SamplerCustomAdvanced",
        "LTXVSeparateAVLatent",
        "LTXVCropGuides",
        "LTXVLatentUpsampler",
        "LatentUpscaleBy",
        "LatentUpscaleModelLoader",
        "VAEDecodeTiled",
        "LTXVAudioVAEDecode",
    }
)

LTX25_CONVROT_TWO_STAGE_AV = ExecutionProfile(
    profile_id="ltx25_convrot_two_stage_av",
    architecture="ltx25",
    source_format="int8_convrot",
    required_assets=(
        "text_encoder",
        "latent_upscaler",
        "video_vae",
        "audio_vae",
    ),
    evidence_node_types=_LTX25_CONVROT_EVIDENCE_NODES,
    required_runtime_nodes=_LTX25_CONVROT_EVIDENCE_NODES
    | frozenset(
        {
            "LTXVPreprocess",
            "EmptyLTXVLatentVideo",
            "LTXVImgToVideoInplace",
            "RandomNoise",
            "CFGGuider",
            "KSamplerSelect",
            "ManualSigmas",
            "CreateVideo",
            "SaveVideo",
        }
    ),
)
execution_profiles.register(LTX25_CONVROT_TWO_STAGE_AV)


def profile_from_explicit_manifest(
    config: Mapping[str, Any],
    *,
    architecture: str,
    source_format: str,
) -> ExecutionProfile | None:
    """Resolve an explicitly named DuckMotion profile from a companion manifest."""
    recipe = config.get("duckmotion_recipe")
    if not isinstance(recipe, Mapping):
        return None
    profile = execution_profiles.get(str(recipe.get("profile") or ""))
    if profile is None:
        return None
    if profile.architecture.lower() != str(architecture or "").lower():
        return None
    if profile.source_format.lower() != str(source_format or "").lower():
        return None
    return profile
