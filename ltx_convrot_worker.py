"""Standalone LTX-2.5 INT8 ConvRot worker.

This worker imports a pinned Comfy core checkout as a Python library. It does not
start ComfyUI, expose a workflow API, or execute the checkpoint's companion JSON
as arbitrary code. The JSON is used only to resolve the required model assets;
execution below is DuckMotion's fixed two-stage LTX recipe.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
from pathlib import Path
import sys
import traceback


HIGH_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
LOW_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
IMAGE_GUIDE_STRENGTH = 0.7


def _snap_dimension(value: int) -> int:
    return max(512, (int(value) // 64) * 64)


def _snap_frames(value: int) -> int:
    value = max(9, int(value))
    return max(9, ((value - 1) // 8) * 8 + 1)


def _result_tuple(value):
    if hasattr(value, "result"):
        result = value.result
        return tuple(result) if isinstance(result, (list, tuple)) else (result,)
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _call_node(nodes_module, node_name: str, **kwargs):
    cls = nodes_module.NODE_CLASS_MAPPINGS.get(node_name)
    if cls is None:
        raise RuntimeError(f"Pinned Comfy core does not provide required node {node_name!r}")
    instance = cls()
    function_name = getattr(cls, "FUNCTION", None)
    if function_name:
        function = getattr(instance, function_name)
    else:
        function = getattr(cls, "execute", None) or getattr(instance, "execute", None)
    if function is None:
        raise RuntimeError(f"Comfy node {node_name!r} has no executable function")

    signature = inspect.signature(function)
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    call_kwargs = kwargs if has_var_kwargs else {key: value for key, value in kwargs.items() if key in signature.parameters}
    value = function(**call_kwargs)
    if inspect.isawaitable(value):
        value = asyncio.run(value)
    return _result_tuple(value)


def _add_asset_folder(folder_paths, category: str, path: str) -> None:
    parent = str(Path(path).expanduser().resolve().parent)
    try:
        folder_paths.add_model_folder_path(category, parent, is_default=True)
    except TypeError:
        folder_paths.add_model_folder_path(category, parent)


def _load_image_tensor(path: str):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    array = np.asarray(image).astype("float32") / 255.0
    return torch.from_numpy(array)[None, ...]


def _save_poster(images, path: Path) -> None:
    import numpy as np
    from PIL import Image

    frame = images[0]
    if hasattr(frame, "detach"):
        frame = frame.detach().float().cpu().numpy()
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255.0).round().astype("uint8")
    Image.fromarray(array[..., :3]).convert("RGB").save(path, quality=92)


def _latest_video(output_dir: Path) -> Path | None:
    candidates = []
    for suffix in ("*.mp4", "*.mkv", "*.webm"):
        candidates.extend(output_dir.rglob(suffix))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _prepare_comfy(output_dir: Path, assets: dict[str, str]):
    comfy_root = Path(str(os.getenv("DUCKMOTION_LTX_CONVROT_COMFY_ROOT") or "")).expanduser()
    if not comfy_root.exists():
        raise RuntimeError(f"Pinned Comfy core checkout is missing: {comfy_root}")
    sys.path.insert(0, str(comfy_root))
    # Comfy parses process argv during import. Worker arguments have already been
    # consumed and must not leak into that parser.
    sys.argv = [sys.argv[0]]

    import folder_paths
    import nodes

    folder_paths.set_output_directory(str(output_dir))
    _add_asset_folder(folder_paths, "diffusion_models", assets["checkpoint"])
    _add_asset_folder(folder_paths, "text_encoders", assets["text_encoder"])
    _add_asset_folder(folder_paths, "latent_upscale_models", assets["latent_upscaler"])
    _add_asset_folder(folder_paths, "vae", assets["video_vae"])
    _add_asset_folder(folder_paths, "vae", assets["audio_vae"])

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    return nodes


def _run(request: dict, output_dir: Path) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("LTX ConvRot runtime currently requires CUDA")

    model_path = str(request["model_path"])
    declared_assets = dict(request.get("assets") or {})
    required = ("text_encoder", "latent_upscaler", "video_vae", "audio_vae")
    missing = [key for key in required if not declared_assets.get(key)]
    if missing:
        raise RuntimeError(f"ConvRot request is missing resolved assets: {', '.join(missing)}")
    assets = {"checkpoint": model_path, **{key: str(declared_assets[key]) for key in required}}

    prompt = str(request.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Prompt is required")
    input_image = str(request.get("input_image") or "").strip() or None
    final_width = _snap_dimension(int(request.get("width") or 1152))
    final_height = _snap_dimension(int(request.get("height") or 768))
    stage1_width = max(256, final_width // 2)
    stage1_height = max(256, final_height // 2)
    num_frames = _snap_frames(int(request.get("num_frames") or 241))
    fps = float(request.get("fps") or 24.0)
    seed = int(request.get("seed") if request.get("seed") is not None else 0)

    output_dir.mkdir(parents=True, exist_ok=True)
    nodes = _prepare_comfy(output_dir, assets)

    model = _call_node(nodes, "UNETLoader", unet_name=Path(model_path).name, weight_dtype="default")[0]
    clip = _call_node(
        nodes,
        "CLIPLoader",
        clip_name=Path(assets["text_encoder"]).name,
        type="ltxv",
        device="default",
    )[0]
    video_vae = _call_node(nodes, "VAELoader", vae_name=Path(assets["video_vae"]).name)[0]
    audio_vae = _call_node(nodes, "LTXVAudioVAELoader", ckpt_name=Path(assets["audio_vae"]).name)[0]
    latent_upscaler = _call_node(
        nodes,
        "LatentUpscaleModelLoader",
        model_name=Path(assets["latent_upscaler"]).name,
    )[0]

    positive = _call_node(nodes, "CLIPTextEncode", clip=clip, text=prompt)[0]
    negative = _call_node(nodes, "CLIPTextEncode", clip=clip, text="")[0]
    conditioned = _call_node(
        nodes,
        "LTXVConditioning",
        positive=positive,
        negative=negative,
        frame_rate=fps,
    )
    positive, negative = conditioned[0], conditioned[1]

    video_latent = _call_node(
        nodes,
        "EmptyLTXVLatentVideo",
        width=stage1_width,
        height=stage1_height,
        length=num_frames,
        batch_size=1,
    )[0]
    if input_image:
        image_tensor = _load_image_tensor(input_image)
        video_latent = _call_node(
            nodes,
            "LTXVImgToVideoInplace",
            vae=video_vae,
            image=image_tensor,
            latent=video_latent,
            strength=IMAGE_GUIDE_STRENGTH,
            bypass=False,
        )[0]

    audio_latent = _call_node(
        nodes,
        "LTXVEmptyLatentAudio",
        frames_number=num_frames,
        frame_rate=fps,
        batch_size=1,
    )[0]
    concat = _call_node(
        nodes,
        "LTXVConcatAVLatent",
        video_latent=video_latent,
        audio_latent=audio_latent,
        model=model,
    )
    av_latent = concat[0]
    if len(concat) > 1:
        model = concat[1]

    noise = _call_node(nodes, "RandomNoise", noise_seed=seed)[0]
    guider = _call_node(
        nodes,
        "CFGGuider",
        model=model,
        positive=positive,
        negative=negative,
        cfg=1.0,
    )[0]
    sampler = _call_node(nodes, "KSamplerSelect", sampler_name="euler")[0]
    high_sigmas = _call_node(nodes, "ManualSigmas", sigmas=HIGH_SIGMAS)[0]
    stage1 = _call_node(
        nodes,
        "SamplerCustomAdvanced",
        noise=noise,
        guider=guider,
        sampler=sampler,
        sigmas=high_sigmas,
        latent_image=av_latent,
    )[0]
    separated = _call_node(nodes, "LTXVSeparateAVLatent", av_latent=stage1, model=model)
    video_latent, audio_latent = separated[0], separated[1]
    if len(separated) > 2:
        model = separated[2]

    video_latent = _call_node(
        nodes,
        "LTXVLatentUpsampler",
        samples=video_latent,
        upscale_model=latent_upscaler,
        vae=video_vae,
    )[0]
    concat2 = _call_node(
        nodes,
        "LTXVConcatAVLatent",
        video_latent=video_latent,
        audio_latent=audio_latent,
        model=model,
    )
    av_latent = concat2[0]
    if len(concat2) > 1:
        model = concat2[1]

    low_sigmas = _call_node(nodes, "ManualSigmas", sigmas=LOW_SIGMAS)[0]
    stage2 = _call_node(
        nodes,
        "SamplerCustomAdvanced",
        noise=_call_node(nodes, "RandomNoise", noise_seed=seed)[0],
        guider=_call_node(
            nodes,
            "CFGGuider",
            model=model,
            positive=positive,
            negative=negative,
            cfg=1.0,
        )[0],
        sampler=sampler,
        sigmas=low_sigmas,
        latent_image=av_latent,
    )[0]
    separated2 = _call_node(nodes, "LTXVSeparateAVLatent", av_latent=stage2, model=model)
    video_latent, audio_latent = separated2[0], separated2[1]

    # VAE objects implement the same decode operation used by Comfy's VAEDecode
    # node. Keeping the decode local avoids depending on an output-node plugin.
    images = video_vae.decode(video_latent["samples"])
    audio = _call_node(
        nodes,
        "LTXVAudioVAEDecode",
        samples=audio_latent,
        audio_vae=audio_vae,
    )[0]

    video = _call_node(
        nodes,
        "CreateVideo",
        images=images,
        fps=fps,
        audio=audio,
        bit_depth=8,
        color_space="sRGB",
    )[0]
    _call_node(
        nodes,
        "SaveVideo",
        video=video,
        filename_prefix="ltx_convrot",
        format={"format": "mp4", "codec": {"codec": "h264", "encoding": {"encoding": "re-encode", "crf": 19}}},
    )
    video_path = _latest_video(output_dir)
    if video_path is None:
        raise RuntimeError("Comfy core completed but did not write an output video")

    poster_path = output_dir / "poster.jpg"
    _save_poster(images, poster_path)
    meta_path = output_dir / "meta.json"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    meta = {
        "plugin": "duckmotion",
        "model": str(request.get("model_name") or Path(model_path).stem),
        "created_at": __import__("time").time(),
        "operation": "image_to_video" if input_image else "text_to_video",
        "prompt": prompt,
        "width": final_width,
        "height": final_height,
        "stage1_width": stage1_width,
        "stage1_height": stage1_height,
        "frame_count": num_frames,
        "fps": fps,
        "seed": seed,
        "audio": True,
        "sampling": "redgraft_convrot_two_stage",
        "high_sigmas": HIGH_SIGMAS,
        "low_sigmas": LOW_SIGMAS,
        "image_guide_strength": IMAGE_GUIDE_STRENGTH if input_image else None,
        "runtime": {
            "device": "cuda",
            "source_format": "int8_convrot",
            "gpu_name": torch.cuda.get_device_name(0),
            "total_vram_gb": round(total_vram_gb, 2),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "video_path": str(video_path),
        "poster_path": str(poster_path),
        "meta_path": str(meta_path),
        "seed": seed,
        "frame_count": num_frames,
        "fps": fps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result_path = Path(args.result)
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        result = _run(request, Path(args.output_dir))
    except Exception as exc:
        result = {
            "ok": False,
            "error": str(exc or exc.__class__.__name__),
            "traceback": traceback.format_exc(),
        }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
