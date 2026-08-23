"""VRAM-aware launcher for the LTX-2.5 ConvRot recipe worker.

The recipe implementation in ``ltx_convrot_worker.py`` stays focused on the
profile's audited node ordering and sampling semantics. This launcher owns
hardware/runtime policy, mirroring WebbDuck's heavyweight Diffusers workers:
choose a conservative policy from VRAM size, keep one-shot caches disabled, and
let the backend's native memory manager offload/slice rather than requiring users
to tune environment variables manually.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
from typing import Any


# Match WebbDuck's job-scoped workers: reduce allocator fragmentation and avoid
# eagerly loading CUDA modules that the selected recipe may never execute.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import ltx_convrot_worker as recipe_worker


LOW_VRAM_GB = 18.0
MID_VRAM_GB = 28.0


def resolve_memory_policy(total_vram_gb: float, mode: str | None = None) -> dict[str, Any]:
    """Return Comfy-native memory policy for the available VRAM class.

    ``auto`` is intentionally conservative on 16 GB-class GPUs. DynamicVRAM is
    the Comfy equivalent of Diffusers submodule/sequential offload; split cross
    attention is the equivalent of attention slicing; the recipe already uses
    tiled spatial/temporal VAE decoding.
    """
    selected = str(
        mode
        if mode is not None
        else os.getenv("DUCKMOTION_LTX_CONVROT_MEMORY", "auto")
    ).strip().lower()

    if selected == "auto":
        if total_vram_gb < LOW_VRAM_GB:
            selected = "conservative"
        elif total_vram_gb < MID_VRAM_GB:
            selected = "balanced"
        else:
            selected = "performance"

    if selected in {"conservative", "low", "lowvram", "sequential"}:
        return {
            "name": "conservative",
            "comfy_args": (
                "--enable-dynamic-vram",
                "--vram-headroom", "1.5",
                "--cache-none",
                "--disable-smart-memory",
                "--use-split-cross-attention",
            ),
            "vram_headroom_gb": 1.5,
            "dynamic_vram": True,
            "aggressive_offload": True,
            "split_attention": True,
        }

    if selected in {"balanced", "model"}:
        return {
            "name": "balanced",
            "comfy_args": (
                "--enable-dynamic-vram",
                "--vram-headroom", "0.75",
                "--cache-none",
            ),
            "vram_headroom_gb": 0.75,
            "dynamic_vram": True,
            "aggressive_offload": False,
            "split_attention": False,
        }

    if selected in {"performance", "none", "off", "unrestricted"}:
        return {
            "name": "performance",
            # The worker is one generation per process, so retaining Comfy node
            # outputs can only increase memory usage and never helps a later job.
            "comfy_args": ("--cache-none",),
            "vram_headroom_gb": 0.0,
            "dynamic_vram": False,
            "aggressive_offload": False,
            "split_attention": False,
        }

    raise ValueError(
        "DUCKMOTION_LTX_CONVROT_MEMORY must be auto, conservative, balanced, or performance"
    )


def _prepare_comfy(output_dir: Path, assets: dict[str, str]):
    """Initialize pinned Comfy with a VRAM-aware policy before its first import."""
    import torch

    comfy_root = Path(str(os.getenv("DUCKMOTION_LTX_CONVROT_COMFY_ROOT") or "")).expanduser()
    if not comfy_root.exists():
        raise RuntimeError(f"Pinned Comfy core checkout is missing: {comfy_root}")

    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free_vram_gb = 0.0
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
        free_vram_gb = free_bytes / 1024**3
    except Exception:
        pass

    policy = resolve_memory_policy(total_vram_gb)
    print(
        "DuckMotion ConvRot memory policy: "
        f"{policy['name']} (VRAM={total_vram_gb:.2f} GiB, "
        f"free_before_load={free_vram_gb:.2f} GiB, "
        f"headroom={policy['vram_headroom_gb']:.2f} GiB)"
    )

    if str(comfy_root) not in sys.path:
        sys.path.insert(0, str(comfy_root))

    # The recipe worker has already consumed its own CLI. Comfy parses argv at
    # import time, so give it only the hardware policy chosen above.
    sys.argv = [sys.argv[0], *policy["comfy_args"]]

    import folder_paths
    import nodes

    folder_paths.set_output_directory(str(output_dir))
    recipe_worker._add_asset_folder(folder_paths, "diffusion_models", assets["checkpoint"])
    recipe_worker._add_asset_folder(folder_paths, "text_encoders", assets["text_encoder"])
    recipe_worker._add_asset_folder(folder_paths, "latent_upscale_models", assets["latent_upscaler"])
    recipe_worker._add_asset_folder(folder_paths, "vae", assets["video_vae"])
    recipe_worker._add_asset_folder(folder_paths, "vae", assets["audio_vae"])

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    return nodes


def main() -> int:
    # Keep profile semantics in one implementation while injecting only the
    # runtime-memory composition point.
    recipe_worker._prepare_comfy = _prepare_comfy
    return recipe_worker.main()


if __name__ == "__main__":
    raise SystemExit(main())
