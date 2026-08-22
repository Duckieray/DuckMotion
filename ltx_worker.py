"""Standalone LTX-2.5 Diffusers worker.

Runs the distilled LTX-2.5 path with its explicit sigma schedule and synchronized
audio. This worker is intentionally isolated because LTX-2.5 currently requires
Diffusers main rather than a released package.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path


def _snap_dimension(value: int) -> int:
    return max(256, (int(value) // 32) * 32)


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


def _run(request: dict, output_dir: Path) -> dict:
    import torch
    from diffusers import LTX2ImageToVideoPipeline, LTX2Pipeline
    from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES
    from diffusers.utils import encode_video, load_image

    if not torch.cuda.is_available():
        raise RuntimeError("LTX-2.5 runtime currently requires a CUDA/ROCm torch device")

    model_path = str(request["model_path"])
    prompt = str(request["prompt"])
    width = _snap_dimension(int(request.get("width") or 768))
    height = _snap_dimension(int(request.get("height") or 512))
    num_frames = _snap_frames(int(request.get("num_frames") or 121))
    fps = float(request.get("fps") or 24.0)
    seed = int(request.get("seed") or 0)
    input_image = request.get("input_image")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipeline_cls = LTX2ImageToVideoPipeline if input_image else LTX2Pipeline
    pipe = pipeline_cls.from_pretrained(model_path, dtype=dtype)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    generator = torch.Generator("cuda").manual_seed(seed)
    kwargs = {
        "prompt": prompt,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "frame_rate": fps,
        "sigmas": DISTILLED_SIGMA_VALUES,
        "guidance_scale": 1.0,
        "audio_guidance_scale": 1.0,
        "stg_scale": 0.0,
        "audio_stg_scale": 0.0,
        "modality_scale": 1.0,
        "audio_modality_scale": 1.0,
        "generator": generator,
        "output_type": "np",
        "return_dict": False,
    }
    if input_image:
        kwargs["image"] = load_image(str(input_image))

    video, audio = pipe(**kwargs)

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
        "width": width,
        "height": height,
        "seed": seed,
        "audio": True,
        "operation": "image_to_video" if input_image else "text_to_video",
        "prompt": prompt,
        "sampling": "distilled",
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
