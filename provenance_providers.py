"""Composition root for DuckMotion's built-in checkpoint provenance providers."""

from __future__ import annotations

from civitai_provenance import resolve_civitai_provenance
from model_provenance import (
    CheckpointProvenanceProvider,
    checkpoint_provenance_providers,
)


def register_builtin_provenance_providers() -> None:
    if "civitai_sha256" not in checkpoint_provenance_providers.ids():
        checkpoint_provenance_providers.register(
            CheckpointProvenanceProvider(
                provider_id="civitai_sha256",
                resolve=resolve_civitai_provenance,
            )
        )
