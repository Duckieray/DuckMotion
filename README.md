# DuckMotion

DuckMotion is a separately managed WebbDuck web plugin that adds a local Wan2.2 image-to-video workspace using a diffusers runtime.

## Current Status

- Local DuckMotion job queue + worker
- WebbDuck-compatible runtime profile reuse (`device` / `dtype` selection)
- Image staging (upload or copy from WebbDuck outputs)
- Configurable output directory with plugin-local video gallery
- Wan2.2 I2V generation path via `diffusers.WanImageToVideoPipeline` (requires compatible diffusers build)

## Repo Layout

```text
DuckMotion/
|- plugin.json
|- backend.py
|- tools/
|  `- install_webbduck_plugin.py
`- ui/
   |- index.html
   |- app.js
   `- styles.css
```

## Install Into WebbDuck

WebbDuck discovers web plugins under:

```text
<plugins-root>/webapps/<plugin-id>/
```

Install DuckMotion into `webapps/duckmotion` by copy/symlink, or use the installer script:

```bash
python3 tools/install_webbduck_plugin.py --webbduck-dir /path/to/webbduck --overwrite
```

Or install into a shared user plugin root:

```bash
python3 tools/install_webbduck_plugin.py --plugins-dir ~/.webbduck/plugins --overwrite
```

## Install Runtime Dependencies (Same Environment as WebbDuck)

DuckMotion runs inside the WebbDuck process, so install DuckMotion requirements into the same Python environment you use for WebbDuck.

After installing WebbDuck's own requirements, install DuckMotion extras:

```bash
pip install -r requirements.txt
```

If your installed `diffusers` build does not include `WanImageToVideoPipeline`, install a newer version (or source build) and re-run the DuckMotion health check.

## Quick Start Model Example (Wan2.2 Diffusers Base)

DuckMotion's current runtime uses the **diffusers Wan2.2 model repo**, not the older GGUF pair/VAE setup.

Recommended model:

- `Wan-AI/Wan2.2-I2V-A14B-Diffusers`

Install/download it like a standard Hugging Face model:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B-Diffusers --local-dir /path/to/Wan2.2-I2V-A14B-Diffusers
```

Recommended local folder convention (keeps Wan models organized next to WebbDuck checkpoints without mixing them with SDXL):

```text
webbduck/checkpoint/wan/Wan2.2-I2V-A14B-Diffusers/
```

Examples:

- Windows: `C:\Users\<you>\path\to\webbduck\checkpoint\wan\Wan2.2-I2V-A14B-Diffusers`
- WSL/Linux: `/path/to/webbduck/checkpoint/wan/Wan2.2-I2V-A14B-Diffusers`

Example download directly into that folder:

```bash
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B-Diffusers --local-dir /path/to/webbduck/checkpoint/wan/Wan2.2-I2V-A14B-Diffusers
```

In DuckMotion Setup, local models are auto-discovered and listed in the model selector.
Discovery scans:

- `webbduck/checkpoint/wan` and `webbduck/checkpoints/wan`
- WebbDuck models-root override paths when `WEBBDUCK_MODELS_DIR` is set (for example `<models>/checkpoint/wan` or `<models>/checkpoints/wan`)
- DuckMotion `Models Cache Dir` (if configured)
- Hugging Face cache roots (`HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` / default cache path)

Default behavior is zero-config friendly:

- If a local Wan model is found in `checkpoint/wan`, DuckMotion auto-selects it.
- If not, DuckMotion falls back to `Wan-AI/Wan2.2-I2V-A14B-Diffusers`.
- Outputs go under WebbDuck outputs by default (`outputs/duckmotion_videos`).

You can then use either:

1. `Wan-AI/Wan2.2-I2V-A14B-Diffusers` in DuckMotion Setup (auto-download on first run), or
2. `/path/to/Wan2.2-I2V-A14B-Diffusers` in DuckMotion Setup (local path from the `huggingface-cli download` command)

Note:

- You do **not** need to separately download a GGUF high/low pair or a standalone VAE for the current diffusers runtime path.

## Configuration

DuckMotion stores plugin state under:

```text
~/.webbduck/plugin_state/
```

Optional environment variables:

- `DUCKMOTION_MODEL_ID_OR_PATH` (or `DUCKMOTION_MODEL_ID`)
- `DUCKMOTION_MODELS_DIR` (optional cache/model dir)
- `DUCKMOTION_OUTPUT_DIR` (plugin video outputs)
- `DUCKMOTION_RUNTIME_BACKEND` (`auto` default, internal backend selector)
- `DUCKMOTION_GGUF_TRANSFORMER_PATH` (optional explicit Wan GGUF transformer file)
- `DUCKMOTION_DEFAULT_WIDTH`
- `DUCKMOTION_DEFAULT_HEIGHT`
- `DUCKMOTION_DEFAULT_FRAMES`
- `DUCKMOTION_DEFAULT_FPS`
- `DUCKMOTION_DEFAULT_STEPS`
- `DUCKMOTION_DEFAULT_GUIDANCE_SCALE`
- `DUCKMOTION_CUDA_MODE` (`offload` default, `full` optional)
- `DUCKMOTION_MEMORY_POLICY` (`auto` default, `off`, `balanced`, or `aggressive`)
- `DUCKMOTION_SAFETY_MODE` (`block` default, `warn`, or `off`)
- `DUCKMOTION_KEEP_PIPELINE_LOADED` (`0` default, set `1` to keep Wan pipeline cached after jobs)

## Runtime + Device Handling

DuckMotion runs inside the WebbDuck process and reuses WebbDuck's runtime profile resolution behavior.
That means DuckMotion follows WebbDuck environment/runtime settings such as:

- `WEBBDUCK_DEVICE`
- `WEBBDUCK_DTYPE`
- `WEBBDUCK_STRICT_DEVICE`

By default, DuckMotion uses CPU offload mode for CUDA and unloads its Wan pipeline after each job to avoid interfering with WebbDuck image generation VRAM usage.

DuckMotion is intended to stay a single application even when multiple internal runtimes are supported. Backend selection is internal:

- `auto` keeps the UI simple and lets DuckMotion choose the safest viable runtime path
- standard diffusers remains the baseline runtime
- when compatible GGUF assets are available, DuckMotion can prefer an internal hybrid diffusers + GGUF path on tighter systems without exposing a second app or workflow graph to users

Before DuckMotion loads Wan, it also performs a best-effort cleanup pass against the in-process WebbDuck runtime:

- unload loaded WebbDuck SDXL pipelines and cached components
- unload captioner models when present
- run Python GC and CUDA cache cleanup before Wan placement

DuckMotion also applies an adaptive memory policy by default:

- `auto` inspects the WebbDuck runtime profile and chooses extra Wan memory reductions based on device mode and available VRAM
- lower-VRAM CUDA systems prefer more aggressive options such as sequential CPU offload, attention slicing, and VAE tiling when those features exist in the installed diffusers build
- higher-VRAM full-CUDA systems avoid extra reductions unless you explicitly request them

If you need to override the automatic choice, set `DUCKMOTION_MEMORY_POLICY` to:

- `off`: disable extra Wan memory reductions
- `balanced`: enable moderate memory reductions with limited performance impact
- `aggressive`: prefer the lowest-VRAM path available, even when it is slower

DuckMotion also applies a Windows-focused preflight safety gate before queueing Wan jobs:

- `block` refuses clearly dangerous jobs before model load begins
- `warn` allows the job but reports detected risk in the API response and job warnings
- `off` disables the guardrail entirely

The safety gate is resource-aware. It checks detected runtime VRAM, requested resolution, frame count, step count, CUDA mode, memory policy, and Windows host-memory/page-file availability when that information is available.

## Requirements

DuckMotion depends on the WebbDuck runtime environment plus a diffusers build that includes:

- `WanImageToVideoPipeline`
- `diffusers.utils.export_to_video`

If the installed diffusers version does not include Wan video support, the DuckMotion health panel will report it.

Video export also needs FFmpeg support. The `requirements.txt` includes `imageio-ffmpeg`; if export still fails on your system, install a system `ffmpeg` binary as well.

## Known Limitations (Current Scaffold)

- Running-job cancellation is best-effort (queued jobs can cancel cleanly; active diffusers jobs may complete before cancellation takes effect)
- This plugin keeps its own job list and gallery (it does not yet extend WebbDuck's global queue/gallery UI)
- Large Wan2.2 models can require significant VRAM and may need additional runtime tuning/offload settings later

## Windows Memory Note (Important for Wan2.2 A14B)

On Windows, large Wan checkpoints can fail to load with:

- `os error 1455`
- `The paging file is too small for this operation to complete`
- `MemoryError` during `from_pretrained`

This is system virtual-memory exhaustion (page file), not just VRAM.

Recommended fix:

1. Open `System Properties` -> `Advanced` -> `Performance Settings` -> `Advanced` -> `Virtual memory`.
2. Use `System managed size`, or set a custom size around `65536` to `131072` MB total.
3. Reboot Windows.
4. Retry DuckMotion generation.

## Development

DuckMotion imports WebbDuck modules at runtime (storage + runtime profile resolution), so run it in a WebbDuck environment.
