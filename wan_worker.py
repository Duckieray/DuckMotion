"""Standalone Wan Diffusers worker.

The worker deliberately owns every Wan-specific dependency and memory decision.
DuckMotion's plugin process only sees the generic VideoBackend contract.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path


def _snap_dimension(value: int, multiple: int = 16) -> int:
    value = max(256, int(value))
    return max(multiple, (value // multiple) * multiple)


def _snap_frames(value: int) -> int:
    """Wan video lengths use 4*k+1 frames."""
    value = max(5, int(value))
    return max(5, ((value - 1) // 4) * 4 + 1)


def _system_memory_gb() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / 1024**3
    except Exception:
        return 0.0


def _source_size_gb(source: str) -> float:
    path = Path(source).expanduser()
    if not path.exists() or not path.is_dir():
        return 0.0
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except OSError:
        return 0.0
    return float(total) / 1024**3


def _load_pipeline(model_path: str, image_to_video: bool, dtype):
    from diffusers import WanImageToVideoPipeline, WanPipeline

    cls = WanImageToVideoPipeline if image_to_video else WanPipeline
    kwargs = {"low_cpu_mem_usage": True}
    try:
        return cls.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError:
        # Older released Diffusers used torch_dtype while main uses dtype.
        return cls.from_pretrained(model_path, torch_dtype=dtype, **kwargs)


def _configure_memory(pipe, device: str, total_vram_gb: float, output_dir: Path) -> str:
    if device != "cuda":
        pipe.to("cpu")
        return "cpu"

    mode = str(os.getenv("DUCKMOTION_WAN_OFFLOAD", "auto")).strip().lower()
    if mode == "auto":
        mode = "group" if total_vram_gb < 24.0 else "model"

    if mode in {"group", "leaf", "leaf_group"}:
        if hasattr(pipe, "enable_group_offload"):
            kwargs = {
                "onload_device": __import__("torch").device("cuda"),
                "offload_device": __import__("torch").device("cpu"),
                "offload_type": "leaf_level",
                "use_stream": False,
                "low_cpu_mem_usage": True,
            }
            source_size = _source_size_gb(str(getattr(pipe, "_name_or_path", "") or ""))
            system_ram = _system_memory_gb()
            if source_size and system_ram and source_size > system_ram * 0.80:
                disk_dir = output_dir / ".wan_offload"
                disk_dir.mkdir(parents=True, exist_ok=True)
                kwargs["offload_to_disk_path"] = str(disk_dir)
            try:
                pipe.enable_group_offload(**kwargs)
                return "group"
            except TypeError:
                # Some Diffusers revisions do not yet expose low_cpu_mem_usage
                # at the pipeline convenience method.
                kwargs.pop("low_cpu_mem_usage", None)
                pipe.enable_group_offload(**kwargs)
                return "group"
        pipe.enable_sequential_cpu_offload(device="cuda")
        return "sequential"

    if mode in {"sequential", "seq"}:
        pipe.enable_sequential_cpu_offload(device="cuda")
        return "sequential"

    if mode in {"model", "cpu"}:
        pipe.enable_model_cpu_offload(device="cuda")
        return "model"

    pipe.to("cuda")
    return "none"


def _save_poster(frames, path: Path) -> None:
    from PIL import Image
    import numpy as np

    frame = frames[0]
    if isinstance(frame, Image.Image):
        frame.convert("RGB").save(path, quality=92)
        return
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255.0).round().astype("uint8")
    Image.fromarray(array).convert("RGB").save(path, quality=92)


def _run(request: dict, output_dir: Path) -> dict:
    import torch
    from diffusers.utils import export_to_video, load_image

    if not torch.cuda.is_available():
        raise RuntimeError("Wan runtime currently requires CUDA")

    model_path = str(request["model_path"])
    prompt = str(request.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Prompt is required")

    input_image = str(request.get("input_image") or "").strip() or None
    negative_prompt = str(request.get("negative_prompt") or "")
    width = _snap_dimension(int(request.get("width") or 832))
    height = _snap_dimension(int(request.get("height") or 480))
    num_frames = _snap_frames(int(request.get("num_frames") or 81))
    fps = int(request.get("fps") or 16)
    steps = max(1, int(request.get("num_inference_steps") or 30))
    guidance = float(request.get("guidance_scale") if request.get("guidance_scale") is not None else 5.0)
    seed = int(request.get("seed") if request.get("seed") is not None else 0)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = _load_pipeline(model_path, bool(input_image), dtype)

    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    output_dir.mkdir(parents=True, exist_ok=True)
    offload = _configure_memory(pipe, "cuda", total_vram_gb, output_dir)

    generator = torch.Generator("cuda").manual_seed(seed)
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": generator,
        "output_type": "pil",
    }
    if input_image:
        kwargs["image"] = load_image(input_image).convert("RGB")

    result = pipe(**kwargs)
    frames = result.frames[0]

    video_path = output_dir / "video.mp4"
    poster_path = output_dir / "poster.jpg"
    meta_path = output_dir / "meta.json"
    export_to_video(frames, str(video_path), fps=fps)
    _save_poster(frames, poster_path)

    meta = {
        "plugin": "duckmotion",
        "model": Path(model_path).name or model_path,
        "created_at": __import__("time").time(),
        "operation": "image_to_video" if input_image else "text_to_video",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "frame_count": num_frames,
        "fps": fps,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "seed": seed,
        "runtime": {
            "device": "cuda",
            "dtype": str(dtype).replace("torch.", ""),
            "offload": offload,
            "total_vram_gb": round(total_vram_gb, 2),
            "system_memory_gb": round(_system_memory_gb(), 2),
            "source_size_gb": round(_source_size_gb(model_path), 2),
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
