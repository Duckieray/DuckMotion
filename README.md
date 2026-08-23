# DuckMotion

DuckMotion is WebbDuck's local, model-driven video generation plugin. The user
selects a model; DuckMotion discovers its capabilities and routes generation to
a compatible runtime without exposing architecture or backend choices in the
normal workflow.

Current runnable video workflows:

- Wan 2.2 pure text-to-video checkpoints through an isolated Diffusers runtime.
- Wan 2.2 pure image-to-video checkpoints through the same isolated runtime.
- Wan 2.2 paired GGUF transformer checkpoints through the same Wan runtime,
  assembled with Diffusers pipeline components behind the model selection.
- Wan 2.2 TI2V-5B text-to-video through Diffusers. The upstream checkpoint also
  supports image-to-video, but current Diffusers does not expose that TI2V image
  path, so DuckMotion intentionally does not advertise it yet.
- LTX-2.5 text-to-video and image-to-video with synchronized audio through an
  isolated two-stage Diffusers runtime.

## Design Rules

- The model/checkpoint is the user-facing selection.
- Architecture, checkpoint format, and backend IDs are internal routing metadata.
- Inputs and controls come from model capabilities, defaults, and constraints.
- A model is only marked runnable when an installed backend owns its workflow.
- Model runtimes are process-isolated from WebbDuck and from each other.
- DuckMotion's host environment does not install Wan/LTX Diffusers stacks.

See `docs/MODEL_DRIVEN_VIDEO_ARCHITECTURE.md` and `AGENTS.md` before changing
runtime or discovery behavior.

## Layout

```text
DuckMotion/
|- plugin_backend.py        # model-driven FastAPI composition root
|- model_runtime.py         # descriptors, capabilities, backend resolver
|- model_discovery.py       # local + Hugging Face cache discovery
|- job_runtime.py           # architecture-neutral job coordinator
|- host_runtime.py          # WebbDuck runtime + GPU lease bridge
|- runtime_services.py      # host/storage composition
|- runtime_surfaces.py      # health/config/status surfaces
|- storage_runtime.py       # config/jobs/staging/gallery persistence
|- storage_api.py
|- wan_backend.py           # isolated Wan adapter
|- wan_worker.py            # Wan Diffusers + GGUF worker
|- ltx_backend.py           # isolated LTX-2.5 adapter
|- ltx_worker.py            # LTX-only two-stage Diffusers process
|- runtime_requirements/
|  |- wan.txt
|  `- ltx25.txt
|- ui/                      # capability-driven browser workspace
`- tests/
```

The former monolithic `backend.py` has been removed.

## Installation Into WebbDuck

WebbDuck discovers web plugins under:

```text
<plugins-root>/webapps/<plugin-id>/
```

Install by copy/symlink, or use:

```bash
python3 tools/install_webbduck_plugin.py --webbduck-dir /path/to/webbduck --overwrite
```

A shared user plugin root also works:

```bash
python3 tools/install_webbduck_plugin.py --plugins-dir ~/.webbduck/plugins --overwrite
```

DuckMotion's top-level `requirements.txt` intentionally contains no video model
engine. WebbDuck supplies the plugin-host web/runtime dependencies.

## Isolated Runtime Environments

### Wan

Create a dedicated Python environment using:

```bash
pip install -r runtime_requirements/wan.txt
```

Install the PyTorch build appropriate for the host CUDA stack, then point
DuckMotion at that interpreter:

```bash
export DUCKMOTION_WAN_PYTHON=/path/to/wan-env/bin/python
```

The Wan runtime supports both normal Diffusers model layouts and the historical
hybrid Diffusers + GGUF path.

- Pure T2V checkpoints run through `WanPipeline`.
- Pure I2V checkpoints run through `WanImageToVideoPipeline` and require a
  source image.
- Local `.gguf` transformer files are discovered as models. For Wan2.2 A14B
  high/low denoiser pairs, names ending in `H.gguf` / `L.gguf` (including
  `_H.gguf` / `_L.gguf`) are grouped into one selectable checkpoint. DuckMotion
  loads both with `WanTransformer3DModel.from_single_file()` and
  `GGUFQuantizationConfig`, then injects them as `transformer` and
  `transformer_2` into the normal Wan pipeline. The UI never asks the user to
  choose a GGUF backend.
- The Diffusers base repo supplies scheduler/VAE/text-encoder/config components
  for GGUF models. DuckMotion selects the normal Wan2.2 A14B T2V or I2V base
  from the requested operation. Advanced deployments can override those
  component sources with `DUCKMOTION_WAN_GGUF_T2V_BASE` and
  `DUCKMOTION_WAN_GGUF_I2V_BASE`.
- `Wan-AI/Wan2.2-TI2V-5B-Diffusers` is recognized as a TI2V checkpoint, but the
  current Diffusers integration exposes its text-conditioned path only. Its
  descriptor records that upstream I2V capability internally while keeping the
  public/runnable `image_to_video` capability false until a runtime actually
  implements it.

On a 16 GB GPU, automatic memory policy prefers group offloading and falls back
to sequential CPU offload when necessary. Very large local checkpoints can use
Diffusers disk-backed group offload when host RAM is insufficient. GGUF reduces
transformer memory/storage pressure but still depends on the base pipeline
components and does not make every generation cheap.

### LTX-2.5

Create a separate environment using:

```bash
pip install -r runtime_requirements/ltx25.txt
```

Then set:

```bash
export DUCKMOTION_LTX_PYTHON=/path/to/ltx-env/bin/python
```

LTX-2.5 currently tracks Diffusers main because its APIs have not yet landed in
a stable Diffusers release.

DuckMotion uses the reference distilled two-stage path:

1. diffusion at half the requested final resolution;
2. 2x latent spatial upsampling;
3. short full-resolution refinement using the stage-2 distilled sigma schedule;
4. synchronized audio/video encoding.

The selected width and height describe the final output. Final dimensions are
therefore normalized to multiples of 64, and frame counts use the LTX `8k+1`
constraint. On a 16 GB GPU, the Diffusers worker defaults to sequential CPU
offload.

## Model Discovery

DuckMotion searches architecture-neutral model roots plus the normal Hugging
Face cache. A model does not need to be copied into a DuckMotion-specific
folder. Local discovery includes recognized Diffusers `model_index.json`
directories and Wan `.gguf` transformer files. Paired H/L GGUF files appear as
one model entry rather than two backend choices.

Relevant environment variables include:

- `WEBBDUCK_MODELS_DIR`
- `WEBBDUCK_HF_CACHE_DIR`
- `HF_HUB_CACHE`
- `HUGGINGFACE_HUB_CACHE`
- `HF_HOME`
- `DUCKMOTION_MODELS_DIR`

The persisted user configuration contains only:

- `model_id_or_path`
- `models_dir`
- `output_dir`

There is no persisted architecture/backend/format selector. Hugging Face cache
selections persist the canonical repository ID. Local Diffusers models and
local GGUF models persist their local source path.

## Capability-Driven UI

The browser workspace consumes the same public model descriptors as the API.
It does not choose a runtime family or checkpoint format.

When the selected model changes, the UI automatically:

- shows or hides source-image staging from `image_to_video`;
- requires a source image only when `source_image_required` is true;
- permits text-only generation when `text_to_video` is true;
- shows negative prompt only when supported;
- applies checkpoint defaults for dimensions, frames, FPS, steps, and guidance;
- applies dimension and frame-count constraints from the descriptor;
- hides arbitrary step/guidance editing when the checkpoint reports a locked
  sampling schedule;
- shows synchronized-audio capability in the model profile;
- disables discovered models whose runtime is unavailable.

Setup no longer stores model-family generation defaults. Those values belong to
the model descriptor and generation request.

## Runtime Environment Variables

Generic:

- `DUCKMOTION_MODEL_ID_OR_PATH`
- `DUCKMOTION_MODELS_DIR`
- `DUCKMOTION_OUTPUT_DIR`

Wan runtime:

- `DUCKMOTION_WAN_PYTHON`
- `DUCKMOTION_WAN_TIMEOUT_SECONDS`
- `DUCKMOTION_WAN_OFFLOAD` (`auto`, `group`, `sequential`, `model`, `none`)
- `DUCKMOTION_WAN_GGUF_T2V_BASE` (optional advanced component-source override)
- `DUCKMOTION_WAN_GGUF_I2V_BASE` (optional advanced component-source override)

LTX runtime:

- `DUCKMOTION_LTX_PYTHON`
- `DUCKMOTION_LTX_TIMEOUT_SECONDS`
- `DUCKMOTION_LTX_OFFLOAD` (`auto`, `sequential`, `model`, `none`)

DuckMotion reuses WebbDuck's runtime-profile and GPU-lease services through the
architecture-neutral `host_runtime.py` bridge. The actual model code executes
in child processes, so worker exit releases model resources.

## Current Model Defaults

Defaults are properties of the detected model, not UI engine presets.

Generic Wan 2.2 defaults currently exposed by the descriptor are 832x480,
81 frames, 16 fps, 30 steps, and guidance 5.0. The published TI2V-5B variant is
checkpoint-specific: 1280x704 landscape, 121 frames, 24 fps, 50 steps, and
guidance 5.0. Checkpoints identified as Turbo receive their fast checkpoint
defaults rather than changing the application mode.

LTX-2.5 defaults describe the final output: 1536x1024, 121 frames, 24 fps, with
the distilled two-stage schedule and guidance 1.0. Its sampling schedule is
locked to the checkpoint's explicit distilled sigma values rather than being a
generic arbitrary-step workflow.

## API Surface

The plugin exposes model-driven routes for:

- model discovery
- config
- health/runtime status
- generation/jobs/cancellation
- staging
- gallery
- recent WebbDuck images

Public model payloads expose capabilities, defaults, constraints, and readiness.
They do not require clients to understand architecture/backend IDs or GGUF.

## Development Status

The runtime architecture and browser UI are model-driven. Real hardware smoke
validation must cover the restored Wan GGUF path alongside normal Wan Diffusers
and LTX-2.5 before the video migration is considered regression-clean.

No full cross-model GPU validation should be inferred from the presence of a
backend. The planned smoke matrix covers Wan Diffusers, Wan GGUF and LTX-2.5
alongside WebbDuck's SDXL, FLUX, Krea, and Qwen image backends.
