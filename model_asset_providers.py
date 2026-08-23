"""Generic support-asset provider registry for DuckMotion model setup.

Providers own architecture/format-specific discovery and recipe inspection.
The setup tool only sees normalized manifests and resolved asset paths, so adding
a new model family does not require another branch in ``tools/setup.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ltx_convrot_assets import (
    convrot_asset_cache_root,
    inspect_convrot_assets,
    is_ltx25_convrot_path,
)


DiscoverFn = Callable[[Path], list[Path]]
InspectFn = Callable[..., dict[str, Any]]
CacheRootFn = Callable[[], Path]


@dataclass(frozen=True)
class ModelAssetProvider:
    provider_id: str
    label: str
    runtime_id: str
    discover: DiscoverFn
    inspect: InspectFn
    cache_root: CacheRootFn


class ModelAssetProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, ModelAssetProvider] = {}

    def register(self, provider: ModelAssetProvider) -> None:
        provider_id = str(provider.provider_id or "").strip()
        if not provider_id:
            raise ValueError("Model asset provider must define provider_id")
        if provider_id in self._providers:
            raise ValueError(f"Duplicate DuckMotion model asset provider: {provider_id}")
        self._providers[provider_id] = provider

    def providers(self) -> tuple[ModelAssetProvider, ...]:
        return tuple(self._providers.values())

    def ids(self) -> tuple[str, ...]:
        return tuple(self._providers.keys())


model_asset_providers = ModelAssetProviderRegistry()


def _discover_ltx25_convrot(models_dir: Path) -> list[Path]:
    try:
        candidates = sorted(models_dir.rglob("*.safetensors"), key=lambda p: str(p).lower())
    except OSError:
        return []
    return [path for path in candidates if is_ltx25_convrot_path(path)]


model_asset_providers.register(
    ModelAssetProvider(
        provider_id="ltx25_convrot",
        label="LTX-2.5 ConvRot",
        runtime_id="ltx25_convrot",
        discover=_discover_ltx25_convrot,
        inspect=inspect_convrot_assets,
        cache_root=convrot_asset_cache_root,
    )
)
