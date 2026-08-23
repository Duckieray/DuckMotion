# DuckMotion

DuckMotion is WebbDuck's local, model-driven video generation plugin. Pick a
model; DuckMotion discovers its capabilities and routes it to the right isolated
runtime. Architecture, checkpoint format, GGUF, ConvRot, Comfy, execution-profile,
and backend IDs are implementation details rather than user-facing modes.

Current runnable workflows include:

- Wan 2.2 text-to-video and image-to-video through an isolated Diffusers runtime;
- Wan 2.2 paired GGUF H/L checkpoints through that same Wan runtime;
- Wan 2.2 TI2V-5B text-to-video where current Diffusers supports it;
- LTX-2.5 two-stage text/image-to-video with synchronized audio;
- LTX-2.5 INT8 ConvRot through recipe-driven execution profiles and a pinned
  Comfy-core library runtime (no ComfyUI server or UI).

## Quick Start

The normal installation path is intentionally one command plus your shared model
folder:

```bash
python tools/setup.py --models /path/to/models
```

`setup.py`:

1. saves the shared model-library path;
2. prepares DuckMotion's isolated Wan/LTX/ConvRot runtimes;
3. repairs runtime-owned adjuncts such as the pinned ConvRot Comfy checkout;
4. prepares support assets declared by recipes for discovered models;
5. uses deterministic runtime locations under
   `~/.local/share/duckmotion/runtimes/`;
6. automatically installs/refreshes the DuckMotion WebbDuck plugin when a
   sibling `../WebbDuck` checkout is found;
7. requires no `DUCKMOTION_*_PYTHON` exports or special model-folder layout.

If WebbDuck lives elsewhere:

```bash
python tools/setup.py \
  --models /path/to/models \
  --webbduck-dir /path/to/WebbDuck
```

Then run the non-loading diagnostic:

```bash
python tools/doctor.py
```

Doctor scans the configured model root **and** the normal Hugging Face cache,
checks the prepared isolated runtimes, and reports per-model readiness/missing
recipe/assets without loading model weights or generating video.

After that, start/restart WebbDuck and use DuckMotion normally.

## Model Library

DuckMotion does not require one proprietary folder layout. Point setup at the
root of an existing shared model library. Discovery recursively recognizes:

- Diffusers `model_index.json` directories;
- Wan `.gguf` files, including H/L dual-transformer pairs;
- supported LTX single-file formats such as INT8 ConvRot;
- cached Hugging Face Diffusers snapshots from the user's ordinary HF cache.

A typical shared root can therefore look like:

```text
models/
├── checkpoints/
│   ├── wan/
│   ├── ltx/
│   └── ...
├── lora/
├── embeddings/
└── ...
```

Wan H/L GGUF files only need to live beside each other. They appear as one
selectable model; DuckMotion finds the mate automatically.

Single-file formats may need a declarative companion recipe. The recipe describes
an execution profile and its support-asset roles; it is not a second user-facing
model selection. DuckMotion resolves declared assets recursively from the shared
model root and its own cache, and setup may fetch missing assets when the recipe
provides a trusted Hugging Face source URL.

## Checkpoint vs. Recipe vs. Display Name

DuckMotion keeps three identities separate:

1. **Checkpoint descriptor** — architecture, weight format, capabilities.
2. **Execution profile** — the runtime recipe, required assets, runtime surface,
   defaults and constraints.
3. **Display/model identity** — the community or vendor model name shown to the
   user.

A name such as REDGraft, Dark Beast, or a future community model never selects
runtime behavior. At most, a legacy filename can be a weak format-discovery hint
when the file lacks useful metadata. Recipe selection still requires an explicit
profile declaration or structural recipe evidence.

The generic flow is:

```text
checkpoint
  -> detect architecture + format
  -> resolve compatible execution profile
  -> resolve profile-declared assets
  -> dispatch profile-owned worker
```

If DuckMotion recognizes a format but cannot resolve a supported recipe, doctor
reports that state instead of guessing.

## Runtime Isolation

DuckMotion owns three default runtime directories:

```text
~/.local/share/duckmotion/runtimes/
├── wan/
├── ltx25/
└── ltx25_convrot/
```

The plugin configures those interpreter paths automatically at startup. This is
the reproducible default path for normal installs.

Advanced deployments may override them with:

- `DUCKMOTION_WAN_PYTHON`
- `DUCKMOTION_LTX_PYTHON`
- `DUCKMOTION_LTX_CONVROT_PYTHON`
- `DUCKMOTION_RUNTIME_HOME`

Those are overrides, **not installation requirements**.

`tools/prepare_model_runtimes.py` remains available for targeted maintenance:

```bash
python tools/prepare_model_runtimes.py wan
python tools/prepare_model_runtimes.py ltx25
python tools/prepare_model_runtimes.py ltx25_convrot
python tools/prepare_model_runtimes.py all
```

It installs libraries/runtime code only and does not download model weights.
`--repair-only` repairs runtime-owned resources without reinstalling packages.

`tools/prepare_model_assets.py` is the generic support-asset setup surface. It
iterates registered asset providers rather than branching on model names.

## Wan

The isolated Wan runtime supports ordinary Diffusers model layouts and hybrid
Diffusers + local GGUF transformers.

For dual-transformer Wan2.2 GGUF checkpoints, names ending in `H.gguf` /
`L.gguf` are grouped as one model. DuckMotion loads both local denoisers with
`WanTransformer3DModel.from_single_file()` + `GGUFQuantizationConfig` and uses
normal Wan Diffusers components for scheduler/VAE/text encoding.

The default component sources are selected from the requested workflow.
Advanced deployments can override them with:

- `DUCKMOTION_WAN_GGUF_T2V_BASE`
- `DUCKMOTION_WAN_GGUF_I2V_BASE`

On lower-VRAM GPUs the runtime uses backend-owned offloading without exposing a
memory/backend selector in the normal UI.

## LTX-2.5

The standard LTX runtime follows the distilled two-stage path:

1. stage-one diffusion at half final resolution;
2. latent spatial upsampling;
3. short full-resolution refinement;
4. synchronized audio/video output.

Final dimensions are normalized to multiples of 64 and frame counts to `8k+1`.
The sampling schedule is checkpoint/runtime-owned rather than a generic
arbitrary-step UI mode.

## LTX-2.5 INT8 ConvRot

ConvRot has a separate isolated runtime because activation rotation needs a
runtime that understands the format. The current implementation imports a pinned
Comfy core as a Python library only. It does not start ComfyUI, expose its
UI/server, call its HTTP API, or execute arbitrary workflow JSON.

ConvRot **format detection does not choose a sampling recipe**. A companion recipe
must resolve to an installed execution profile. DuckMotion currently ships the
`ltx25_convrot_two_stage_av` profile, whose implementation was audited against a
known REDGraft workflow. REDGraft is therefore an important compatibility/test
fixture, not the architectural routing key.

A future unrelated ConvRot checkpoint can use the same profile by declaring it,
or a different profile can be registered with its own worker/defaults/assets
without creating a new backend or UI mode.

DuckMotion accepts either:

- a small explicit `duckmotion_recipe` manifest; or
- a supported exported-workflow shape that an adapter can map to a known profile.

The companion is declarative evidence only. See `docs/LTX25_CONVROT.md` for the
profile and manifest contract.

## Configuration

Normal persisted user configuration is deliberately small:

- `model_id_or_path`
- `models_dir`
- `output_dir`

There is no persisted architecture, backend, GGUF, ConvRot, Comfy, runtime, or
execution-profile selector.

Useful generic advanced environment overrides include:

- `DUCKMOTION_MODEL_ID_OR_PATH`
- `DUCKMOTION_MODELS_DIR`
- `DUCKMOTION_OUTPUT_DIR`
- `WEBBDUCK_HF_CACHE_DIR`
- `HF_HUB_CACHE`
- `HUGGINGFACE_HUB_CACHE`
- `HF_HOME`

## Capability-Driven UI

The DuckMotion browser consumes public model capabilities/defaults/constraints.
It automatically:

- permits T2V only when supported;
- shows source-image staging only for I2V-capable models;
- requires an image only when the selected checkpoint requires one;
- shows negative prompts only when supported;
- applies model-specific dimensions/frames/FPS/steps/guidance;
- applies dimension/frame constraints;
- hides arbitrary sampling controls for locked schedules;
- reports synchronized-audio capability;
- disables discovered models that are not runnable.

Users select checkpoints, not engines or recipes.

## Architecture Rules

- Model/checkpoint identity is the user-facing routing key.
- Architecture, format, backend and recipe/profile IDs stay private.
- Checkpoint-format detection must not choose recipe-specific defaults.
- Generic API/job/storage/UI/setup code must not branch on model brands.
- Backend-specific dependencies live in isolated runtimes/workers.
- Recipe-specific assets, defaults, runtime nodes and worker entrypoints live in
  execution profiles/providers.
- A model is marked runnable only when its format, recipe, assets and runtime are
  all ready.
- Adding another model/runtime should extend discovery + a provider/profile or
  backend adapter, not add another application mode.

See `AGENTS.md` and `docs/MODEL_DRIVEN_VIDEO_ARCHITECTURE.md` before changing
runtime/discovery behavior.

## Development / Validation

Hardware validation is intentionally API-driven and separate from setup:

```bash
python tools/run_hardware_smoke.py
```

The default is preflight only. Actual generation requires explicit `--execute`;
heavy/reference rows additionally require `--include-heavy`.

The smoke harness selects ConvRot targets from runtime format metadata rather
than model names. See `docs/HARDWARE_SMOKE_MATRIX.md` for the current matrix.

## Plugin Packaging

For manual/advanced plugin installation, the lower-level installer remains:

```bash
python tools/install_webbduck_plugin.py \
  --webbduck-dir /path/to/WebbDuck \
  --overwrite
```

Normal users should prefer `tools/setup.py`, which invokes it automatically when
possible.
