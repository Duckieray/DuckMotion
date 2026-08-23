# DuckMotion

DuckMotion is WebbDuck's local, model-driven video generation plugin. Pick a
model; DuckMotion discovers its capabilities and routes it to the right isolated
runtime. Architecture, checkpoint format, GGUF, ConvRot, Comfy, provenance,
execution-profile, and backend IDs are implementation details rather than
user-facing modes.

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
4. prepares trusted checkpoint provenance, companion recipes, and support assets
   when a discovered format needs them;
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
checks the prepared isolated runtimes, and reports per-model readiness, cached
provenance, recipe state, and missing assets without loading model weights,
contacting provenance services, or generating video.

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
model root and its own cache, and setup may fetch missing assets from trusted
recipe/profile sources.

When a recognized checkpoint format has no local companion recipe, setup can use
registered provenance providers to identify the **exact file by strong content
hash** and cache published JSON sidecars. The first built-in provider uses
Civitai's public SHA256 model-version lookup. Once cached, doctor/readiness use the
local provenance record only. DuckMotion does not re-download the selected model.

## Checkpoint vs. Provenance vs. Recipe vs. Display Name

DuckMotion keeps four identities separate:

1. **Checkpoint descriptor** — architecture, weight format, capabilities.
2. **Checkpoint provenance** — strong content fingerprint and canonical source
   identity for that exact file.
3. **Execution profile** — the runtime recipe, required assets, runtime surface,
   defaults and constraints.
4. **Display/model identity** — the community or vendor model name shown to the
   user.

A name such as REDGraft, Dark Beast, or a future community model never selects
runtime behavior. At most, a legacy filename can be a weak format-discovery hint
when the file lacks useful metadata. Provenance names/IDs are diagnostic only.
Recipe selection still requires an explicit profile declaration or structural
recipe evidence.

The generic flow is:

```text
checkpoint
  -> detect architecture + format
  -> resolve trusted provenance/recipe evidence when needed
  -> resolve compatible execution profile
  -> resolve profile-declared assets
  -> dispatch profile-owned worker
```

If DuckMotion recognizes a format but cannot resolve trusted provenance or a
supported recipe, doctor reports that state instead of guessing.

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

`tools/prepare_model_assets.py` is the generic provenance/recipe/support-asset
setup surface. It iterates registered providers rather than branching on model
names. Strong checkpoint fingerprints and provenance records are cached under
`~/.cache/duckmotion/provenance/`; prepared support assets use DuckMotion's asset
cache.

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

If that companion is not beside the checkpoint, setup may recover a published
sidecar through trusted hash provenance. A provenance match still cannot choose
the profile by model/version name; the cached JSON must pass the same recipe
adapter as a local companion.

The installed two-stage AV profile also owns trusted standard sources for its
LTX text encoder, VAE, and spatial-upscaler roles. Those defaults only fill a
missing source for the same standard asset name; a differently named custom
recipe asset is never silently substituted.

See `docs/LTX25_CONVROT.md` and `docs/EXECUTION_RECIPES.md` for the profile,
provenance, and manifest contracts.

## Configuration

Normal persisted user configuration is deliberately small:

- `model_id_or_path`
- `models_dir`
- `output_dir`

There is no persisted architecture, backend, GGUF, ConvRot, Comfy, runtime,
provenance-provider, or execution-profile selector.

Useful generic advanced environment overrides include:

- `DUCKMOTION_MODEL_ID_OR_PATH`
- `DUCKMOTION_MODELS_DIR`
- `DUCKMOTION_OUTPUT_DIR`
- `DUCKMOTION_PROVENANCE_CACHE`
- `DUCKMOTION_ASSET_CACHE`
- `WEBBDUCK_HF_CACHE_DIR`
- `HF_HUB_CACHE`
- `HUGGINGFACE_HUB_CACHE`
- `HF_HOME`

`CIVITAI_API_TOKEN` is an optional advanced credential used only when Civitai
requires authenticated access to a provenance sidecar. Public SHA256 provenance
lookup does not require it.

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

Users select checkpoints, not engines, provenance sources, or recipes.

## Architecture Rules

- Model/checkpoint identity is the user-facing routing key.
- Architecture, format, backend, provenance-provider and recipe/profile IDs stay
  private.
- Checkpoint-format detection must not choose recipe-specific defaults.
- Provenance identifies the exact file; provenance names/IDs must not choose a
  recipe.
- Generic API/job/storage/UI/setup code must not branch on model brands.
- Backend-specific dependencies live in isolated runtimes/workers.
- Recipe-specific assets, defaults, runtime nodes and worker entrypoints live in
  execution profiles/providers.
- A model is marked runnable only when its format, recipe, assets and runtime are
  all ready.
- Adding another model/runtime should extend discovery + a provenance/asset
  provider/profile or backend adapter, not add another application mode.

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
