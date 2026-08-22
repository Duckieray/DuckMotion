"""Standalone LTX-2.5 Diffusers worker.

Runs the reference distilled two-stage LTX-2.5 path:
1. stage-1 diffusion at half the requested final resolution;
2. latent 2x spatial upsampling;
3. short stage-2 refinement at final resolution with synchronized audio.

The worker is isolated because LTX-2.5 currently requires Diffusers main rather
than a released package and has very different memory requirements from Wan.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path


def _snap_final_dimension(value: int) -> int:
    # Final output is halved for stage 1; stage 1 must still be divisible by 32.
    return max(512, (int(value) // 64) * 64)


def _snap_frames(value: int) -> int:
    value = max(9, int(value))
    return max(9, ((value - 1) // 8) * 8 + 1)


def _save_poster(video, path: Path) -> None:
    import numpy as np
    from PIL import Image

    frame = np.asarray(video[0])
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).round().astype("uint8")
    Image.fromarray(frame).convert("RGB").save(path, quality=92)


def _configure_main_pipeline(pipe, device: str, total_vram_gb: float) -> str:
    if device != "cuda":
        pipe.to("cpu")
        return "cpu"

    mode = str(os.getenv("DUCKMOTION_LTX_OFFLOAD", "auto")).strip().lower()
    if mode == "auto":
        # The distilled 22B transformer plus Gemma 4 encoder cannot reside on
        # a 16 GB card. Sequential offload is slower but is the safe default.
        mode = "sequential" if total_vram_gb < 32.0 else "model"

    if mode in {"sequential", "seq"}:
        pipe.enable_sequential_cpu_offload(device="cuda")
        return "sequential"
    if mode in {"model", "cpu"}:
        pipe.enable_model_cpu_offload(device="cuda")
        return "model"
    pipe.to("cuda")
    return "none"


def _run(request: dict, output_dir: Path) -> dict:
    import torch
    from diffusers import (
        LTX2ImageToVideoPipeline,
        LTX2LatentUpsamplePipeline,
        LTX2Pipeline,
    )
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.pipelines.ltx2.utils import (
        DEFAULT_NEGATIVE_PROMPT,
        DISTILLED_SIGMA_VALUES,
        STAGE_2_DISTILLED_SIGMA_VALUES,
    )
    from diffusers.utils import encode_video, load_image

    if not torch.cuda.is_available():
        raise RuntimeError("LTX-2.5 runtime currently requires CUDA")

    model_path = str(request["model_path"])
    prompt = str(request.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Prompt is required")

    final_width = _snap_final_dimension(int(request.get("width") or 1536))
    final_height = _snap_final_dimension(int(request.get("height") or 1024))
    stage1_width = final_width // 2
    stage1_height = final_height // 2
    num_frames = _snap_frames(int(request.get("num_frames") or 121))
    fps = float(request.get("fps") or 24.0)
    seed = int(request.get("seed") if request.get("seed") is not None else 0)
    input_image = str(request.get("input_image") or "").strip() or None

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipeline_cls = LTX2ImageToVideoPipeline if input_image else LTX2Pipeline
    try:
        pipe = pipeline_cls.from_pretrained(model_path, dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:
        pipe = pipeline_cls.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)

    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    offload = _configure_main_pipeline(pipe, "cuda", total_vram_gb)
    pipe.vae.enable_tiling()

    latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        model_path,
        subfolder="latent_upsampler",
        dtype=dtype,
    )
    upsample_pipe = LTX2LatentUpsamplePipeline(
        vae=pipe.vae,
        latent_upsampler=latent_upsampler,
    )
    try:
        upsample_pipe.enable_model_cpu_offload(device="cuda")
        upsample_offload = "model"
    except Exception:
        latent_upsampler.to("cuda")
        upsample_offload = "cuda"

    generator = torch.Generator("cuda").manual_seed(seed)
    shared = {
        "prompt": prompt,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "frame_rate": fps,
        "guidance_scale": 1.0,
        "audio_guidance_scale": 1.0,
        "stg_scale": 0.0,
        "audio_stg_scale": 0.0,
        "modality_scale": 1.0,
        "audio_modality_scale": 1.0,
        "generator": generator,
        "return_dict": False,
    }
    if input_image:
        shared["image"] = load_image(input_image).convert("RGB")

    # Stage 1: coherent low-resolution video/audio latents using the checkpoint's
    # explicit distilled sigma schedule. Do not replace this with a generic
    # num_inference_steps schedule; that silently reduces quality.
    video_latent, audio_latent = pipe(
        width=stage1_width,
        height=stage1_height,
        num_frames=num_frames,
        sigmas=DISTILLED_SIGMA_VALUES,
        output_type="latent",
        **shared,
    )

    upsample_kwargs = {
        "latents": video_latent,
        "output_type": "latent",
        "return_dict": False,
    }
    try:
        upscaled_video_latent = upsample_pipe(
            latents_normalized=False,
            **upsample_kwargs,
        )[0]
    except TypeError:
        # Older LTX2 Diffusers revisions inferred latent normalization.
        upscaled_video_latent = upsample_pipe(**upsample_kwargs)[0]

    # Stage 2: short full-resolution refinement, continuing the same generator
    # stream and the audio latents produced in stage 1.
    video, audio = pipe(
        latents=upscaled_video_latent,
        audio_latents=audio_latent,
        num_frames=num_frames,
        sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
        noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
        output_type="np",
        **shared,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "video.mp4"
    poster_path = output_dir / "poster.jpg"
    meta_path = output_dir / "meta.json"

    encode_video(
        video[0],
        fps=int(round(fps)),
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        output_path=str(video_path),
    )
    _save_poster(video[0], poster_path)

    meta = {
        "plugin": "duckmotion",
        "model": "LTX-2.5",
        "created_at": __import__("time").time(),
        "frame_count": num_frames,
        "fps": fps,
        "width": final_width,
        "height": final_height,
        "stage1_width": stage1_width,
        "stage1_height": stage1_height,
        "seed": seed,
        "audio": True,
        "operation": "image_to_video" if input_image else "text_to_video",
        "prompt": prompt,
        "sampling": "distilled_two_stage",
        "stage1_sigma_count": len(DISTILLED_SIGMA_VALUES),
        "stage2_sigma_count": len(STAGE_2_DISTILLED_SIGMA_VALUES),
        "runtime": {
            "device": "cuda",
            "dtype": str(dtype).replace("torch.", ""),
            "offload": offload,
            "upsampler_offload": upsample_offload,
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
