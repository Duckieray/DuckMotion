# Model-Driven Video Architecture

Status: **runtime and capability-driven UI implemented; real-model smoke validation pending**

DuckMotion follows the same checkpoint-first rule as WebbDuck:

> The user selects a model. DuckMotion determines capabilities, readiness,
> runtime, trusted provenance when needed, execution recipe, and support assets
> automatically.

Architecture/backend/provenance/profile identifiers are internal routing metadata.
They are never required user choices.

## Implemented Flow

```text
selected model
    |
    v
model discovery / checkpoint descriptor
    |
    +-- architecture + source format (private)
    +-- runnable capabilities
    +-- architecture-level defaults/constraints
    |
    +--> optional setup-time provenance resolution
    |       |
    |       +-- strong checkpoint fingerprint
    |       +-- trusted provider identity
    |       `-- cached declarative recipe candidates
    |
    v
capability-driven browser + generic API
    |
    v
VideoJobCoordinator
    |
    v
VideoBackendResolver
    |
    +--> isolated Wan worker
    +--> isolated LTX-2.5 Diffusers worker
    `--> LTX-2.5 ConvRot backend
             |
             v
       execution profile registry
             |
             +-- required assets
             +-- trusted standard asset sources
             +-- required runtime surface
             +-- profile defaults/constraints
             `-- worker entrypoint
    |
    v
generic storage / jobs / gallery
```

The former monolithic Wan `backend.py` has been removed. Generic runtime,
router, storage, job, UI, and setup code do not dispatch on model brands.

## Product Contract

Normal generation must not ask the user to choose:

- architecture family;
- Diffusers/reference runtime implementation;
- quantization/loading implementation;
- checkpoint provenance provider;
- execution profile/recipe;
- in-process/subprocess mode;
- pipeline class names.

A public capability means the installed implementation can execute that workflow
once its runtime/recipe/assets are ready. Detection metadata may retain upstream
capabilities that are not yet runnable, but the UI must not advertise them as
working controls.

## Public Model Descriptor

The browser consumes architecture-free model payloads:

```json
{
  "name": "...",
  "source": "...",
  "capabilities": {
    "text_to_video": true,
    "image_to_video": false,
    "video_to_video": false,
    "audio_output": false,
    "negative_prompt": true,
    "source_image_required": false
  },
  "defaults": {},
  "constraints": {},
  "supported": true
}
```

Architecture, source-format, backend, provenance-provider, and execution-profile
IDs remain internal.

## Discovery and Identity

`model_discovery.py` scans architecture-neutral local roots plus the normal
Hugging Face cache. Canonical repository identity participates in detection so a
snapshot directory named only by a revision hash does not erase checkpoint
variant information.

The UI persists:

- canonical `repo_id` for Hugging Face cache models;
- local filesystem path for local models.

For single-file formats, lightweight header metadata and compatible sidecar
recipe evidence can participate in format detection. Filename hints are only
compatibility fallbacks; a display/model brand must never select provenance or a
recipe.

## Checkpoint Descriptor vs. Provenance vs. Execution Recipe

`VideoModelDescriptor` describes the checkpoint itself:

- architecture;
- source/weight format;
- capabilities;
- architecture/checkpoint-level defaults and constraints;
- preferred backend.

It must **not** inject defaults merely because one known model using that format
happens to recommend them.

`CheckpointProvenanceProvider` answers where an exact local checkpoint came from.
Setup may fingerprint a local file with SHA256 and query registered trusted
providers. A successful provider can return canonical source metadata and cache
small declarative recipe candidates.

Provenance is not recipe selection. Provider model/version names, IDs, URLs,
tags, and descriptions are diagnostics only. Exactly one provider must match; if
zero or multiple providers match, DuckMotion leaves the model unresolved.

`ExecutionProfile` describes a supported runtime recipe:

- compatible architecture + source format;
- worker entrypoint;
- required support-asset roles;
- trusted standard asset sources;
- structural evidence used by recipe adapters;
- required runtime node/API surface;
- recipe-specific defaults and constraints.

This distinction is important for quantized/community formats. Two checkpoints
can share `ltx25/int8_convrot` while requiring different sampling recipes.
DuckMotion may recognize both checkpoint formats while only one has a currently
installed profile.

The trust chain is intentionally one-way:

```text
strong file identity -> provenance -> declarative recipe evidence -> profile
```

If zero profiles match a recipe, the model remains blocked. If multiple profiles
or multiple supported recipe candidates match, DuckMotion refuses to guess.

## Provenance and Recipe Inputs

A companion recipe is declarative evidence only; it is never arbitrary executable
workflow code.

Two recipe adapters are currently supported for ConvRot:

1. native `duckmotion_recipe` JSON explicitly naming an installed profile and
   its asset roles;
2. supported exported-workflow shapes mapped structurally to an installed
   profile by node/model evidence.

When no usable local companion exists, setup may consult registered provenance
providers. The first built-in provider resolves Civitai model versions by full
SHA256, verifies the returned version includes an exact matching SHA256 file, and
caches only small JSON sidecars. The selected checkpoint is never downloaded or
replaced.

Once a recipe candidate is cached, doctor and runtime readiness use it locally;
they do not perform provenance network calls or re-hash a large unchanged
checkpoint.

The execution profile, not the provenance provider or workflow file, owns the
worker implementation.

## Capability-Driven Browser

The browser uses the unified model catalog as its only model selector.

When a model changes, the UI automatically:

- shows source-image staging only when `image_to_video` is runnable;
- requires that source only when `source_image_required` is true;
- allows text-only generation when `text_to_video` is true;
- shows negative prompt only when supported;
- applies model defaults;
- applies dimension/frame constraints;
- hides arbitrary step/guidance editing for locked sampling schedules;
- shows audio-output capability;
- disables models whose runtime/recipe/assets are unavailable.

No model-family generation defaults are persisted in Setup. The persisted config
is intentionally small:

```json
{
  "model_id_or_path": "...",
  "models_dir": "...",
  "output_dir": "..."
}
```

## Runtime Ownership

### Generic layers

- `model_runtime.py`: checkpoint descriptors, capabilities, backend resolver;
- `model_discovery.py`: architecture-neutral discovery/introspection;
- `model_provenance.py`: strong fingerprinting, provenance contracts and local cache;
- `provenance_providers.py`: provenance-provider composition;
- `model_recipes.py`: execution-profile contracts and registry;
- `model_asset_providers.py`: normalized support-asset provider registry;
- `job_runtime.py`: request normalization, job lifecycle, GPU lease ownership;
- `host_runtime.py`: WebbDuck runtime profile and GPU lease bridge only;
- `storage_runtime.py`: config/jobs/staging/gallery;
- `runtime_surfaces.py`: health/config/status;
- `plugin_backend.py`: generic API composition;
- `tools/setup.py`: generic orchestration of runtime, provenance/assets, and plugin setup.

These layers must not branch on community/vendor model names.

### Isolated runtimes/workers

Architecture-specific ML packages are not installed in WebbDuck's interpreter.
Normal installs use deterministic runtime interpreters under
`~/.local/share/duckmotion/runtimes/`. Advanced overrides remain available:

- `DUCKMOTION_WAN_PYTHON`;
- `DUCKMOTION_LTX_PYTHON`;
- `DUCKMOTION_LTX_CONVROT_PYTHON`.

The ConvRot environment also owns a pinned Comfy core checkout. DuckMotion imports
that checkout as a Python library only; it does not launch a Comfy server, expose
Comfy's UI, or submit arbitrary workflow JSON over HTTP.

Runtime requirements live under `runtime_requirements/`.

## Asset Providers

`tools/prepare_model_assets.py` is provider-driven. It iterates
`model_asset_providers` and only understands a normalized contract:

- discovered checkpoint/source;
- resolved execution profile;
- declared asset roles;
- already-resolved paths;
- optional trusted recipe/profile source URLs;
- provider-owned asset cache.

The tool itself does not contain an LTX/REDGraft branch. Future model families can
register providers without adding another setup mode.

Support assets may live anywhere under configured/inferred model roots or the
provider cache. Special Comfy-style folder layouts are not a user requirement.
An execution profile may supply a trusted standard asset source only when the
recipe omits that role or names the same standard asset. A differently named
custom asset is never silently substituted.

## Wan

The current Diffusers backend exposes only workflows it can actually run:

- pure text-to-video through `WanPipeline`;
- pure image-to-video through `WanImageToVideoPipeline`.

Pure I2V requires a source image.

### TI2V-5B

The upstream TI2V-5B checkpoint supports T2V and I2V, but current Diffusers
`WanPipeline` exposes only the text-conditioned path for this checkpoint.
DuckMotion therefore publicly reports T2V only while retaining internal
detection metadata that upstream I2V exists.

Published checkpoint defaults are model-specific:

- 1280x704;
- 121 frames;
- 24 fps;
- 50 steps;
- guidance 5.0.

A future runtime that implements TI2V image conditioning can flip the runnable
capability without changing generic API or UI code.

Wan dimensions use a 16-pixel grid and frame counts use `4k+1`.

## LTX-2.5

LTX-2.5 publicly exposes:

- text-to-video;
- image-to-video with optional source image;
- synchronized audio output.

The standard isolated worker follows the distilled two-stage path:

1. half-resolution stage-1 diffusion;
2. 2x latent spatial upsampling;
3. full-resolution stage-2 refinement;
4. synchronized audio/video encoding.

Standard Diffusers descriptor defaults remain:

- 1536x1024;
- 121 frames;
- 24 fps;
- final dimensions divisible by 64;
- frame count `8k+1`.

### LTX-2.5 INT8 ConvRot

ConvRot single-file checkpoints are detected privately and routed to
`ltx25_convrot`; the public model payload remains architecture/backend-free.
Format detection does **not** select a sampling recipe.

The current installed execution profile is:

```text
ltx25_convrot_two_stage_av
```

It owns:

- its worker entrypoint;
- required pinned-Comfy nodes;
- text encoder / latent upscaler / video VAE / audio VAE asset roles and trusted
  standard sources;
- 1152x768 / 241-frame / 24-fps defaults;
- the fixed two-stage AV sampling, guide, upscale, decode, and output contract.

That implementation was audited from a REDGraft workflow, making REDGraft a
reference fixture rather than a runtime identity. A future unrelated ConvRot
checkpoint can reuse this profile or register another compatible profile.

See `docs/LTX25_CONVROT.md` and `docs/EXECUTION_RECIPES.md` for the provenance,
recipe/manifest, and asset contracts.

## Extension Rule

A new video checkpoint should extend the **smallest responsible layer**:

- new checkpoint variant, existing runtime behavior: descriptor/detection only;
- new provenance catalog: register a `CheckpointProvenanceProvider`;
- new recipe for an existing format/backend: add an `ExecutionProfile` and
  worker, plus asset-provider adaptation if needed;
- new support-asset source/layout: add/extend an asset provider;
- genuinely new runtime family/API: add a backend + isolated runtime.

It must not require a new runtime-family tab, model-brand branch, or setup mode in
generic router/job/storage/UI code.

## Validation Milestone

Contract tests cover model discovery, public capabilities, resolver/readiness,
generic job normalization, process isolation, memory policy, storage, the
capability-driven UI contract, provenance caching/ambiguity, exact-SHA provider
validation, execution-profile ambiguity, recipe adapters, asset-provider
iteration, and the current ConvRot two-stage AV primitives.

Real-model GPU validation is still intentionally separate. Backend registration,
asset readiness, or passing unit/contract tests must not be described as a
successful hardware smoke test.
