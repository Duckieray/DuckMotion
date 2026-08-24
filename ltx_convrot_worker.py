"""Worker for DuckMotion's LTX-2.5 ConvRot two-stage AV execution profile.

This worker imports a pinned Comfy core checkout as a Python library. It does not
start ComfyUI, expose a workflow API, or execute the checkpoint's companion JSON
as arbitrary code. Recipe resolution happens before worker dispatch; this module
implements only the registered ``ltx25_convrot_two_stage_av`` topology while
honoring the normalized, allow-listed tuning values supplied by its companion
recipe.
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

from ltx_convrot_recipe import (
    DEFAULT_CFG,
    DEFAULT_IMAGE_GUIDE_STRENGTH,
    DEFAULT_SAMPLER,
    DEFAULT_STAGE1_SIGMAS,
    DEFAULT_STAGE2_NOISE_POLICY,
    DEFAULT_STAGE2_SIGMAS,
    DEFAULT_UPSCALED_IMAGE_GUIDE_STRENGTH,
    normalize_sampler_name,
)


EXECUTION_PROFILE_ID = "ltx25_convrot_two_stage_av"
# Backward-compatible names kept for tests/docs; these are fallback profile
# values now, not immutable checkpoint behavior.
HIGH_SIGMAS = DEFAULT_STAGE1_SIGMAS
LOW_SIGMAS = DEFAULT_STAGE2_SIGMAS
IMAGE_GUIDE_STRENGTH = DEFAULT_IMAGE_GUIDE_STRENGTH
UPSCALED_IMAGE_GUIDE_STRENGTH = DEFAULT_UPSCALED_IMAGE_GUIDE_STRENGTH
IMAGE_PREPROCESS_LONG_EDGE = 1536
IMAGE_PREPROCESS_COMPRESSION = 18
LATENT_UPSCALE_METHOD = "bicubic"
LATENT_UPSCALE_SCALE = 0.5
VIDEO_DECODE_TILE_SIZE = 480
VIDEO_DECODE_OVERLAP = 96
VIDEO_DECODE_TEMPORAL_SIZE = 96
VIDEO_DECODE_TEMPORAL_OVERLAP = 24
VIDEO_OUTPUT_CRF = 16
UINT64_MASK = (1 << 64) - 1


def _snap_dimension(value: int) -> int:
    return max(512, (int(value) // 64) * 64)


def _snap_frames(value: int) -> int:
    value = max(9, int(value))
    return max(9, ((value - 1) // 8) * 8 + 1)


def _stage2_seed(seed: int) -> int:
    """Derive the legacy independent second noise stream reproducibly."""

    return (int(seed) + 1) & UINT64_MASK


def _effective_recipe(request: dict) -> dict:
    raw = request.get("execution_recipe")
    raw = raw if isinstance(raw, dict) else {}

    sampler = normalize_sampler_name(raw.get("sampler")) or DEFAULT_SAMPLER
    stage1_sigmas = str(raw.get("stage1_sigmas") or DEFAULT_STAGE1_SIGMAS).strip()
    stage2_sigmas = str(raw.get("stage2_sigmas") or DEFAULT_STAGE2_SIGMAS).strip()
    try:
        cfg = float(raw.get("cfg", DEFAULT_CFG))
    except (TypeError, ValueError):
        cfg = DEFAULT_CFG
    if not 0.0 <= cfg <= 20.0:
        cfg = DEFAULT_CFG

    def strength(name: str, fallback: float) -> float:
        try:
            value = float(raw.get(name, fallback))
        except (TypeError, ValueError):
            return fallback
        return value if 0.0 <= value <= 1.0 else fallback

    stage2_noise_policy = str(
        raw.get("stage2_noise_policy") or DEFAULT_STAGE2_NOISE_POLICY
    ).strip().lower()
    if stage2_noise_policy not in {"increment", "same_seed"}:
        stage2_noise_policy = DEFAULT_STAGE2_NOISE_POLICY

    return {
        "sampler": sampler,
        "stage1_sigmas": stage1_sigmas,
        "stage2_sigmas": stage2_sigmas,
        "cfg": cfg,
        "image_guide_strength": strength(
            "image_guide_strength", DEFAULT_IMAGE_GUIDE_STRENGTH
        ),
        "upscaled_image_guide_strength": strength(
            "upscaled_image_guide_strength",
            DEFAULT_UPSCALED_IMAGE_GUIDE_STRENGTH,
        ),
        "stage2_noise_policy": stage2_noise_policy,
        "origin": str(raw.get("origin") or "profile_default"),
    }


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
        function = getattr(instance, "execute", None) or getattr(cls, "execute", None)
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


def _load_image_tensor(path: str, *, longer_edge: int = IMAGE_PREPROCESS_LONG_EDGE):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    width, height = image.size
    if width > height:
        resized = (longer_edge, int(height * (longer_edge / width)))
    else:
        resized = (int(width * (longer_edge / height)), longer_edge)
    image = image.resize(resized, Image.Resampling.LANCZOS)
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

    profile_id = str(request.get("execution_profile") or "").strip()
    if profile_id != EXECUTION_PROFILE_ID:
        raise RuntimeError(
            f"Worker implements execution profile '{EXECUTION_PROFILE_ID}', "
            f"not {profile_id or '<missing>'!r}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("LTX ConvRot runtime currently requires CUDA")

    model_path = str(request["model_path"])
    declared_assets = dict(request.get("assets") or {})
    required = ("text_encoder", "latent_upscaler", "video_vae", "audio_vae")
    missing = [key for key in required if not declared_assets.get(key)]
    if missing:
        raise RuntimeError(f"ConvRot request is missing resolved assets: {', '.join(missing)}")
    assets = {"checkpoint": model_path, **{key: str(declared_assets[key]) for key in required}}
    recipe = _effective_recipe(request)

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
    seed = int(request.get("seed") if request.get("seed") is not None else 0) & UINT64_MASK
    stage2_seed = seed if recipe["stage2_noise_policy"] == "same_seed" else _stage2_seed(seed)

    print(
        "DuckMotion ConvRot execution recipe: "
        f"origin={recipe['origin']}; sampler={recipe['sampler']}; cfg={recipe['cfg']}; "
        f"i2v={recipe['image_guide_strength']}/{recipe['upscaled_image_guide_strength']}; "
        f"stage2_noise={recipe['stage2_noise_policy']}"
    )
    print(
        "DuckMotion ConvRot quality assets: "
        f"video_vae={Path(assets['video_vae']).name}; "
        f"upscaler={Path(assets['latent_upscaler']).name}; "
        f"policy={request.get('asset_policy') or 'recipe'}"
    )

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
    # The two-stage AV profile loads both LTX VAEs through core VAELoader. The
    # audio-only loader searches a different category and is not this profile's contract.
    audio_vae = _call_node(nodes, "VAELoader", vae_name=Path(assets["audio_vae"]).name)[0]
    latent_upscaler = _call_node(
        nodes,
        "LatentUpscaleModelLoader",
        model_name=Path(assets["latent_upscaler"]).name,
    )[0]

    positive = _call_node(nodes, "CLIPTextEncode", clip=clip, text=prompt)[0]
    # This profile zeroes positive conditioning rather than encoding an empty
    # negative prompt, preserving the expected LTX conditioning shape.
    negative = _call_node(nodes, "ConditioningZeroOut", conditioning=positive)[0]
    positive, negative = _call_node(
        nodes,
        "LTXVConditioning",
        positive=positive,
        negative=negative,
        frame_rate=fps,
    )[:2]

    image_tensor = None
    if input_image:
        image_tensor = _load_image_tensor(input_image)
        image_tensor = _call_node(
            nodes,
            "LTXVPreprocess",
            image=image_tensor,
            img_compression=IMAGE_PREPROCESS_COMPRESSION,
        )[0]

    video_latent = _call_node(
        nodes,
        "EmptyLTXVLatentVideo",
        width=stage1_width,
        height=stage1_height,
        length=num_frames,
        batch_size=1,
    )[0]
    if image_tensor is not None:
        video_latent = _call_node(
            nodes,
            "LTXVImgToVideoInplace",
            vae=video_vae,
            image=image_tensor,
            latent=video_latent,
            strength=recipe["image_guide_strength"],
            bypass=False,
        )[0]

    audio_latent = _call_node(
        nodes,
        "LTXVEmptyLatentAudio",
        audio_vae=audio_vae,
        frames_number=num_frames,
        frame_rate=fps,
        batch_size=1,
    )[0]
    av_latent = _call_node(
        nodes,
        "LTXVConcatAVLatent",
        video_latent=video_latent,
        audio_latent=audio_latent,
    )[0]

    noise = _call_node(nodes, "RandomNoise", noise_seed=seed)[0]
    guider = _call_node(
        nodes,
        "CFGGuider",
        model=model,
        positive=positive,
        negative=negative,
        cfg=recipe["cfg"],
    )[0]
    sampler = _call_node(nodes, "KSamplerSelect", sampler_name=recipe["sampler"])[0]
    high_sigmas = _call_node(nodes, "ManualSigmas", sigmas=recipe["stage1_sigmas"])[0]
    stage1 = _call_node(
        nodes,
        "SamplerCustomAdvanced",
        noise=noise,
        guider=guider,
        sampler=sampler,
        sigmas=high_sigmas,
        latent_image=av_latent,
    )[0]
    video_latent, audio_latent = _call_node(nodes, "LTXVSeparateAVLatent", av_latent=stage1)[:2]

    # Crop guide frames/conditioning before the second pass. For T2V this is
    # effectively a no-op; for I2V it removes stage-one guide frames so the
    # high-resolution guide can be reapplied after latent upscaling.
    stage2_positive, stage2_negative, video_latent = _call_node(
        nodes,
        "LTXVCropGuides",
        positive=positive,
        negative=negative,
        latent=video_latent,
    )[:3]
    video_latent = _call_node(
        nodes,
        "LTXVLatentUpsampler",
        samples=video_latent,
        upscale_model=latent_upscaler,
        vae=video_vae,
    )[0]
    video_latent = _call_node(
        nodes,
        "LatentUpscaleBy",
        samples=video_latent,
        upscale_method=LATENT_UPSCALE_METHOD,
        scale_by=LATENT_UPSCALE_SCALE,
    )[0]
    if image_tensor is not None:
        video_latent = _call_node(
            nodes,
            "LTXVImgToVideoInplace",
            vae=video_vae,
            image=image_tensor,
            latent=video_latent,
            strength=recipe["upscaled_image_guide_strength"],
            bypass=False,
        )[0]

    av_latent = _call_node(
        nodes,
        "LTXVConcatAVLatent",
        video_latent=video_latent,
        audio_latent=audio_latent,
    )[0]

    low_sigmas = _call_node(nodes, "ManualSigmas", sigmas=recipe["stage2_sigmas"])[0]
    stage2 = _call_node(
        nodes,
        "SamplerCustomAdvanced",
        noise=_call_node(nodes, "RandomNoise", noise_seed=stage2_seed)[0],
        guider=_call_node(
            nodes,
            "CFGGuider",
            model=model,
            positive=stage2_positive,
            negative=stage2_negative,
            cfg=recipe["cfg"],
        )[0],
        sampler=sampler,
        sigmas=low_sigmas,
        latent_image=av_latent,
    )[0]
    video_latent, audio_latent = _call_node(nodes, "LTXVSeparateAVLatent", av_latent=stage2)[:2]

    # The profile uses tiled video VAE decoding with explicit spatial/temporal
    # tile sizes to avoid a large full-latent decode allocation.
    images = _call_node(
        nodes,
        "VAEDecodeTiled",
        samples=video_latent,
        vae=video_vae,
        tile_size=VIDEO_DECODE_TILE_SIZE,
        overlap=VIDEO_DECODE_OVERLAP,
        temporal_size=VIDEO_DECODE_TEMPORAL_SIZE,
        temporal_overlap=VIDEO_DECODE_TEMPORAL_OVERLAP,
    )[0]
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
    # Pinned Comfy core SaveVideo provides the profile's H.264/yuv420p output;
    # no custom output-node package is required.
    _call_node(
        nodes,
        "SaveVideo",
        video=video,
        filename_prefix="ltx_convrot",
        format={"format": "mp4", "codec": {"codec": "h264", "encoding": {"encoding": "re-encode", "crf": VIDEO_OUTPUT_CRF}}},
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
        "stage2_seed": stage2_seed,
        "audio": True,
        "execution_profile": EXECUTION_PROFILE_ID,
        "execution_recipe": recipe,
        "asset_policy": request.get("asset_policy"),
        "quality_upgrades": request.get("quality_upgrades") or {},
        "video_vae": Path(assets["video_vae"]).name,
        "latent_upscaler": Path(assets["latent_upscaler"]).name,
        "sampling": "ltx25_convrot_two_stage_av",
        "sampler": recipe["sampler"],
        "cfg": recipe["cfg"],
        "high_sigmas": recipe["stage1_sigmas"],
        "low_sigmas": recipe["stage2_sigmas"],
        "stage2_noise_policy": recipe["stage2_noise_policy"],
        "image_guide_strength": recipe["image_guide_strength"] if input_image else None,
        "stage2_image_guide_strength": recipe["upscaled_image_guide_strength"] if input_image else None,
        "image_preprocess_long_edge": IMAGE_PREPROCESS_LONG_EDGE if input_image else None,
        "image_preprocess_compression": IMAGE_PREPROCESS_COMPRESSION if input_image else None,
        "latent_upscale_method": LATENT_UPSCALE_METHOD,
        "latent_upscale_scale": LATENT_UPSCALE_SCALE,
        "video_decode_tile_size": VIDEO_DECODE_TILE_SIZE,
        "video_decode_overlap": VIDEO_DECODE_OVERLAP,
        "video_decode_temporal_size": VIDEO_DECODE_TEMPORAL_SIZE,
        "video_decode_temporal_overlap": VIDEO_DECODE_TEMPORAL_OVERLAP,
        "video_output_crf": VIDEO_OUTPUT_CRF,
        "runtime": {
            "device": "cuda",
            "source_format": "int8_convrot",
            "execution_profile": EXECUTION_PROFILE_ID,
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
        "execution_profile": EXECUTION_PROFILE_ID,
        "execution_recipe": recipe,
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
