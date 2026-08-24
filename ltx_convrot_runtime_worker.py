"""VRAM-aware launcher for the LTX-2.5 ConvRot recipe worker.

The recipe implementation in ``ltx_convrot_worker.py`` stays focused on the
profile's audited node ordering and sampling semantics. This launcher owns
hardware/runtime policy, mirroring WebbDuck's heavyweight Diffusers workers:
choose a conservative policy from VRAM size, keep one-shot caches disabled, and
let the backend's native memory manager offload/slice rather than requiring users
to tune environment variables manually.

Comfy is embedded as a Python library here rather than launched through its
``main.py`` entrypoint. That means the launcher must explicitly perform the small
subset of Comfy startup that activates CLI-selected DynamicVRAM. Parsing flags
alone is insufficient: ``comfy-aimdo`` must be initialized, devices registered,
and ``CoreModelPatcher`` switched to the dynamic implementation before model
objects are created.

Normal Comfy prompt execution keeps the full graph lifetime inside one continuous
``torch.inference_mode()`` context. The embedded worker dispatches nodes directly,
so the launcher mirrors that execution contract around the complete recipe run.
A per-node inference context is not equivalent: sampler outputs, lazy hooks, and
model-management work may survive after a node returns and before the next node
is entered.
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
            # DynamicVRAM is enabled by default by current Comfy on supported
            # NVIDIA hardware. Explicitly disable it for the unrestricted tier.
            # The worker is one generation per process, so retaining Comfy node
            # outputs can only increase memory usage and never helps a later job.
            "comfy_args": ("--disable-dynamic-vram", "--cache-none"),
            "vram_headroom_gb": 0.0,
            "dynamic_vram": False,
            "aggressive_offload": False,
            "split_attention": False,
        }

    raise ValueError(
        "DUCKMOTION_LTX_CONVROT_MEMORY must be auto, conservative, balanced, or performance"
    )


def _initialize_aimdo_control():
    """Mirror Comfy's pre-node DynamicVRAM controller initialization.

    ``main.py`` enables argument parsing before importing ``comfy.cli_args`` and
    initializes the global comfy-aimdo controller before importing the node/model
    stack. Embedded users do not get either side effect automatically.
    """
    import comfy.options

    comfy.options.enable_args_parsing()
    from comfy.cli_args import args, enables_dynamic_vram
    import comfy_aimdo.control as aimdo_control

    if enables_dynamic_vram():
        simple_vram_headroom = (
            None if args.reserve_vram is None else int(args.reserve_vram * 1024**3)
        )
        try:
            aimdo_control.init(
                simple_vram_headroom=simple_vram_headroom,
                nvml_pressure=not args.disable_nvml_pressure,
            )
        except TypeError:
            # Support the older comfy-aimdo protocols accepted by the pinned
            # Comfy entrypoint as well.
            try:
                aimdo_control.init(simple_vram_headroom=simple_vram_headroom)
            except TypeError:
                aimdo_control.init()

    return args, enables_dynamic_vram, aimdo_control


def _activate_dynamic_vram(args, enables_dynamic_vram, aimdo_control) -> bool:
    """Register devices and install Comfy's dynamic model patcher.

    This is the second half of the DynamicVRAM startup normally performed by
    Comfy ``main.py`` after ``nodes`` imports its model-management modules.
    """
    import comfy.memory_management as memory_management
    import comfy.model_management as model_management
    import comfy.model_patcher as model_patcher

    supported = bool(model_management.is_nvidia())
    if model_management.is_amd():
        supported = getattr(model_management, "rocm_version", (0, 0)) >= (7, 14)

    requested = bool(args.enable_dynamic_vram) or (
        enables_dynamic_vram() and supported
    )
    if not requested:
        return False

    # Match the pinned Comfy guard. DuckMotion's prepared ConvRot runtime uses
    # torch 2.11, but keep this fail-safe for advanced runtime overrides.
    if (
        not args.enable_dynamic_vram
        and getattr(model_management, "torch_version_numeric", (0, 0)) < (2, 8)
    ):
        return False

    device_headroom = int(float(args.vram_headroom) * 1024**3)
    try:
        initialized = aimdo_control.init_devices(
            (device.index, device_headroom)
            for device in model_management.get_all_torch_devices()
        )
    except TypeError:
        # comfy-aimdo 0.4.9 protocol.
        initialized = aimdo_control.init_devices(
            device.index for device in model_management.get_all_torch_devices()
        )

    if not initialized:
        return False

    model_patcher.CoreModelPatcher = model_patcher.ModelPatcherDynamic
    memory_management.aimdo_enabled = True
    return True


def _prepare_comfy(output_dir: Path, assets: dict[str, str]):
    """Initialize pinned Comfy with a VRAM-aware policy before model loading."""
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
    # import time, so give it only the hardware policy chosen above. Crucially,
    # enable Comfy argument parsing before importing folder_paths/nodes: otherwise
    # comfy.cli_args silently parses [] and every memory flag is ignored.
    sys.argv = [sys.argv[0], *policy["comfy_args"]]
    args, enables_dynamic_vram, aimdo_control = _initialize_aimdo_control()

    import folder_paths
    import nodes

    dynamic_vram_active = _activate_dynamic_vram(
        args,
        enables_dynamic_vram,
        aimdo_control,
    )
    if policy["dynamic_vram"] and not dynamic_vram_active:
        raise RuntimeError(
            "The selected ConvRot memory policy requires Comfy DynamicVRAM, "
            "but the embedded Comfy memory manager did not activate."
        )

    print(
        "DuckMotion ConvRot Comfy memory manager: "
        f"dynamic_vram={'active' if dynamic_vram_active else 'disabled'}; "
        f"split_attention={bool(args.use_split_cross_attention)}; "
        f"aggressive_offload={bool(args.disable_smart_memory)}"
    )

    folder_paths.set_output_directory(str(output_dir))
    recipe_worker._add_asset_folder(folder_paths, "diffusion_models", assets["checkpoint"])
    recipe_worker._add_asset_folder(folder_paths, "text_encoders", assets["text_encoder"])
    recipe_worker._add_asset_folder(folder_paths, "latent_upscale_models", assets["latent_upscaler"])
    recipe_worker._add_asset_folder(folder_paths, "vae", assets["video_vae"])
    recipe_worker._add_asset_folder(folder_paths, "vae", assets["audio_vae"])

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    return nodes


def _install_full_inference_run_context() -> None:
    """Keep the complete embedded Comfy recipe run inside InferenceMode.

    Pinned Comfy's ``PromptExecutor`` enters one ``torch.inference_mode()``
    context around graph execution. A per-node wrapper is not equivalent because
    inference tensors and model-management hooks can remain live between direct
    node calls. Wrap DuckMotion's recipe ``_run`` instead so loading,
    conditioning, both samplers, latent transforms, decode, and output share one
    uninterrupted inference-mode lifetime.
    """
    original_run = recipe_worker._run
    if getattr(original_run, "_duckmotion_full_inference_run", False):
        return

    def run_inference(request, output_dir):
        import torch

        with torch.inference_mode():
            return original_run(request, output_dir)

    run_inference._duckmotion_full_inference_run = True
    recipe_worker._run = run_inference


def main() -> int:
    # Keep profile semantics in one implementation while injecting only the
    # embedded-Comfy runtime composition points.
    recipe_worker._prepare_comfy = _prepare_comfy
    _install_full_inference_run_context()
    return recipe_worker.main()


if __name__ == "__main__":
    raise SystemExit(main())
