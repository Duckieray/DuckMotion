"""Architecture-neutral execution recipe contracts for DuckMotion.

A model descriptor answers *what the checkpoint is* (architecture/format and
capabilities). An execution recipe answers *how a supported runtime should run
it*. Product/display names never participate in recipe selection.

Recipes are declarative contracts. A companion file may name a profile directly,
or an adapter may infer a profile from trusted structural and semantic evidence
in an exported workflow. DuckMotion never executes arbitrary recipe/workflow
code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ExecutionProfile:
    profile_id: str
    architecture: str
    source_format: str
    worker: str
    required_assets: tuple[str, ...] = ()
    # Trusted profile-level defaults are only used when a compatible recipe omits
    # a standard asset declaration/source. They are execution-profile semantics,
    # never checkpoint-brand metadata.
    asset_defaults: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    evidence_node_types: frozenset[str] = frozenset()
    # Distinctive literals can be required when node structure alone is not
    # enough to identify a recipe. LTX-2.5 ConvRot deliberately leaves this
    # empty because sampler/sigma values are checkpoint-author tuning and are
    # normalized separately instead of being used as profile identity.
    evidence_literals: frozenset[str] = frozenset()
    required_runtime_nodes: frozenset[str] = frozenset()
    defaults: Mapping[str, Any] = field(default_factory=dict)
    constraints: Mapping[str, Any] = field(default_factory=dict)


class ExecutionProfileRegistry:
    def __init__(self) -> None:
        self._profiles: dict[str, ExecutionProfile] = {}

    def register(self, profile: ExecutionProfile) -> None:
        profile_id = str(profile.profile_id or "").strip()
        if not profile_id:
            raise ValueError("Execution profile must define profile_id")
        if not str(profile.worker or "").strip():
            raise ValueError(f"Execution profile '{profile_id}' must define a worker")
        if profile_id in self._profiles:
            raise ValueError(f"Duplicate DuckMotion execution profile: {profile_id}")
        unknown_defaults = set(profile.asset_defaults).difference(profile.required_assets)
        if unknown_defaults:
            raise ValueError(
                f"Execution profile '{profile_id}' declares defaults for unknown asset roles: "
                + ", ".join(sorted(unknown_defaults))
            )
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
        """Return a unique structural candidate; adapters validate semantics next."""
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
    worker="ltx_convrot_v3_runtime_worker.py",
    required_assets=(
        "text_encoder",
        "latent_upscaler",
        "video_vae",
        "audio_vae",
    ),
    asset_defaults={
        "text_encoder": {
            "name": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
            "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
            "directory": "text_encoders",
        },
        "latent_upscaler": {
            "name": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            "directory": "latent_upscale_models",
        },
        "video_vae": {
            "name": "ltx-2.5-video-vae-bf16.safetensors",
            "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-bf16.safetensors",
            "directory": "vae",
        },
        "audio_vae": {
            "name": "ltx-2.5-audio-vae-bf16.safetensors",
            "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors",
            "directory": "vae",
        },
    },
    evidence_node_types=_LTX25_CONVROT_EVIDENCE_NODES,
    # Sampling literals are intentionally not profile identity. Current community
    # checkpoints ship their own custom sigmas and may use Euler ancestral while
    # still implementing this exact two-stage AV topology.
    evidence_literals=frozenset(),
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
    defaults={
        "width": 1152,
        "height": 768,
        "num_frames": 241,
        "fps": 24,
        # Eight first-pass transitions plus three refinement transitions.
        "num_inference_steps": 11,
        "guidance_scale": 1.0,
    },
    constraints={
        "dimension_multiple": 64,
        "frame_count_modulo": 8,
        "frame_count_remainder": 1,
        "generation_stages": 2,
        "sampling_schedule_locked": True,
    },
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
