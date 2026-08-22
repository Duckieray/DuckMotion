# Model-Driven Video Architecture

Status: **runtime architecture implemented; capability-driven UI cutover pending**

DuckMotion follows the same product rule as WebbDuck's architecture-agnostic
image generation design:

> The user selects a model. DuckMotion determines capabilities, defaults,
> constraints, and runtime backend automatically.

Architecture/backend identifiers are internal routing metadata and are not
required user choices.

## Implemented Runtime Flow

```text
selected model
    |
    v
model discovery / descriptor
    |
    +-- capabilities
    +-- defaults
    +-- constraints
    +-- readiness
    |
    v
generic API + VideoJobCoordinator
    |
    v
VideoBackendResolver
    |
    +--> isolated Wan backend/worker
    `--> isolated LTX-2.5 backend/worker
    |
    v
generic storage/gallery artifacts
```

The former monolithic Wan `backend.py` has been removed. No generic runtime,
route, storage, or job code depends on a Wan implementation module.

## Product Contract

Normal generation must not ask the user to choose:

- Wan vs LTX vs another architecture;
- Diffusers vs a reference runtime;
- quantization/loading implementation;
- in-process vs subprocess execution;
- pipeline class names.

The model profile tells the UI whether source images are required, optional, or
unsupported and which output/control semantics exist.

A public capability means **the currently installed backend can execute that
workflow**. Detection metadata may retain additional upstream/model capabilities
that are not yet runnable, but the UI must not offer them as working controls.

## Core Ownership

### Model discovery and descriptors

`model_discovery.py` scans architecture-neutral local roots and the normal
Hugging Face cache. `model_runtime.py` produces `VideoModelDescriptor` objects
with public:

- capabilities;
- defaults;
- constraints;
- readiness/support status.

Architecture and backend IDs remain internal.

Discovery evidence should prefer authoritative model/config metadata. Canonical
repo/model identity is included when a Hugging Face snapshot directory itself is
only a revision hash.

### Backend resolver

`VideoBackendResolver` maps descriptors to installed backend adapters. Generic
API/job code never branches on family names.

Model switching calls resolver-wide resource cleanup rather than a
family-specific unload function.

### Job coordination

`VideoJobCoordinator` owns:

- model-capability request validation;
- generic parameter normalization from descriptor defaults/constraints;
- persisted job lifecycle;
- shared WebbDuck GPU lease ownership;
- backend invocation;
- normalized artifact metadata.

It does not know Wan or LTX architecture IDs.

### Host integration

`host_runtime.py` is the only bridge for WebbDuck runtime-profile and GPU-lease
services. It contains no video architecture dependencies.

### Storage

`VideoStorageRuntime` owns config, jobs, staging, output directories, and gallery
persistence. Persisted user configuration is model-neutral:

```json
{
  "model_id_or_path": "...",
  "models_dir": "...",
  "output_dir": "..."
}
```

## Isolated Runtime Contract

Architecture-specific ML packages are not installed in WebbDuck's interpreter.
Each backend launches a dedicated worker interpreter selected by environment:

- `DUCKMOTION_WAN_PYTHON`
- `DUCKMOTION_LTX_PYTHON`

The parent process retains job/GPU-lease ownership. Worker exit releases model
resources.

Runtime dependency sets live under `runtime_requirements/`.

## Wan

The current Diffusers backend owns the workflows it can actually execute:

- pure Wan text-to-video through `WanPipeline`;
- pure Wan image-to-video through `WanImageToVideoPipeline`.

A pure I2V checkpoint marks source image as required.

### Wan2.2 TI2V-5B

The upstream TI2V-5B checkpoint supports both T2V and I2V, but current Diffusers
`WanPipeline` exposes only text conditioning for this checkpoint. DuckMotion
therefore advertises:

```json
{
  "text_to_video": true,
  "image_to_video": false,
  "source_image_required": false
}
```

Internal detection metadata records that upstream I2V exists and that the
current runtime does not implement it. When a native/unified TI2V runtime is
added later, the public capability can change without altering the generic API
or UI architecture.

TI2V-5B has checkpoint-specific published defaults rather than inheriting the
generic Wan I2V profile:

- 1280x704 landscape;
- 121 frames;
- 24 fps;
- 50 inference steps;
- guidance 5.0.

Turbo variants may override the fast sampling defaults while preserving
checkpoint-specific dimensions/frame behavior.

### Wan constraints and memory

Wan constraints currently include:

- dimensions divisible by 16;
- frame count `4k+1`.

The worker uses current Diffusers `WanPipeline` / `WanImageToVideoPipeline` and
owns its memory strategy. On constrained GPUs, automatic placement prefers group
offloading and can fall back to sequential CPU offload. Large local model packs
can request disk-backed group offload when their size clearly exceeds practical
host-memory headroom.

This is a safety/placement mechanism, not a claim that every A14B BF16 package is
practical on a 16 GB GPU. That requires real hardware validation.

## LTX-2.5

LTX-2.5 descriptors expose:

- text-to-video;
- image-to-video;
- synchronized audio output;
- optional source image.

The production worker follows the reference distilled two-stage recipe:

1. stage-1 diffusion at half final resolution with `DISTILLED_SIGMA_VALUES`;
2. 2x latent spatial upsampling with `LTX2LatentUpsamplerModel`;
3. full-resolution refinement with `STAGE_2_DISTILLED_SIGMA_VALUES`;
4. synchronized audio/video encoding.

Consequently:

- selected width/height mean **final output size**;
- final dimensions are divisible by 64 so half-resolution stage 1 remains valid;
- frame count follows `8k+1`;
- `sampling_schedule_locked` is true;
- the distilled sigma schedules are model semantics, not arbitrary UI step
  schedules.

LTX currently tracks Diffusers main in its isolated environment. On a 16 GB GPU,
the worker defaults to sequential CPU offload.

## Public API Principle

Public model payloads should expose enough information for a capability-driven
client while hiding implementation choices:

```json
{
  "name": "...",
  "source": "...",
  "capabilities": {},
  "defaults": {},
  "constraints": {},
  "supported": true
}
```

Clients must not need an architecture/backend selector to generate.

## Remaining UI Cutover

The runtime architecture is model-driven, but the browser UI may still contain
older Wan-shaped presentation. The final UI pass should:

1. use the unified model catalog as the only model selector;
2. render source-image staging only when the model supports/needs it;
3. allow text-only generation when `text_to_video` is true;
4. render negative-prompt controls only when supported;
5. apply model defaults/constraint granularity automatically;
6. hide/lock arbitrary sampling controls when `sampling_schedule_locked` is true;
7. show audio/output capabilities without exposing backend names;
8. remove any remaining Wan-specific labels or engine settings.

No additional runtime family tabs should be introduced during that rewrite.

## Future Extension Rules

A new video architecture or alternative runtime should normally require:

1. descriptor detection/capability/default/constraint rules;
2. a backend adapter;
3. an isolated worker/runtime requirements when dependencies differ;
4. focused tests;
5. zero generic router/job/storage family branching.

A new backend may also unlock a workflow already known in detection metadata,
such as TI2V-5B image conditioning, without changing the generic public contract.

Unknown assets must fail with detection/readiness diagnostics rather than being
guessed as Wan or another existing architecture.

## Validation

Contract/unit tests cover discovery, descriptors, resolver/readiness, generic job
normalization, worker serialization, memory/offload choices, storage surfaces,
and the permanent deletion of the legacy backend module.

Real-model GPU smoke tests remain a separate milestone. Do not infer hardware
success from backend registration alone.
