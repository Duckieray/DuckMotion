# Model-Driven Video Architecture

Status: **runtime and capability-driven UI implemented; real-model smoke validation pending**

DuckMotion follows the same checkpoint-first rule as WebbDuck:

> The user selects a model. DuckMotion determines capabilities, defaults,
> constraints, readiness, and runtime automatically.

Architecture/backend identifiers are internal routing metadata. They are never
required user choices.

## Implemented Flow

```text
selected model
    |
    v
model discovery / descriptor
    |
    +-- runnable capabilities
    +-- defaults
    +-- constraints
    +-- readiness
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
    `--> isolated LTX-2.5 worker
    |
    v
generic storage / jobs / gallery
```

The former monolithic Wan `backend.py` has been removed. Generic runtime,
router, storage, job, and UI code do not dispatch on video architecture names.

## Product Contract

Normal generation must not ask the user to choose:

- architecture family;
- Diffusers/reference runtime implementation;
- quantization/loading implementation;
- in-process/subprocess mode;
- pipeline class names.

A public capability means the **currently installed backend can execute that
workflow**. Detection metadata may retain upstream capabilities that are not yet
runnable, but the UI must not advertise them as working controls.

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

Architecture and backend IDs remain internal.

## Discovery and Identity

`model_discovery.py` scans architecture-neutral local roots plus the normal
Hugging Face cache. Canonical repository identity participates in detection so a
snapshot directory named only by a revision hash does not erase checkpoint
variant information.

The UI persists:

- canonical `repo_id` for Hugging Face cache models;
- local filesystem path for local models.

That keeps checkpoint-specific defaults and capabilities stable across reloads.

## Capability-Driven Browser

The browser uses the unified model catalog as its only model selector.

When a model changes, the UI automatically:

- shows source-image staging only when `image_to_video` is runnable;
- requires that source only when `source_image_required` is true;
- allows text-only generation when `text_to_video` is true;
- shows negative prompt only when supported;
- applies model defaults;
- applies `dimension_multiple` and frame modulo/remainder constraints;
- hides arbitrary step/guidance editing for locked sampling schedules;
- shows audio-output capability;
- disables models whose runtime is unavailable.

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

- `model_runtime.py`: descriptors, capabilities, backend resolver;
- `model_discovery.py`: discovery/introspection;
- `job_runtime.py`: request normalization, job lifecycle, GPU lease ownership;
- `host_runtime.py`: WebbDuck runtime profile and GPU lease bridge only;
- `storage_runtime.py`: config/jobs/staging/gallery;
- `runtime_surfaces.py`: health/config/status;
- `plugin_backend.py`: generic API composition.

These layers must not import architecture-specific pipeline classes.

### Isolated workers

Architecture-specific ML packages are not installed in WebbDuck's interpreter.
Backends launch dedicated worker interpreters selected by:

- `DUCKMOTION_WAN_PYTHON`;
- `DUCKMOTION_LTX_PYTHON`.

Runtime requirements live under `runtime_requirements/`.

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

The worker follows the distilled two-stage path:

1. half-resolution stage-1 diffusion;
2. 2x latent spatial upsampling;
3. full-resolution stage-2 refinement;
4. synchronized audio/video encoding.

Descriptor semantics therefore describe final output:

- default 1536x1024;
- 121 frames;
- 24 fps;
- final dimensions divisible by 64;
- frame count `8k+1`;
- `generation_stages = 2`;
- `sampling_schedule_locked = true`.

The browser hides arbitrary step/guidance editing for this locked schedule.

## Extension Rule

A new video model/runtime should normally require only:

1. detection and descriptor metadata;
2. a backend adapter;
3. an isolated worker/runtime environment when needed;
4. focused tests.

It must not require a new runtime-family tab or architecture branch in generic
router/job/storage/UI code.

## Validation Milestone

Contract tests cover model discovery, public capabilities, resolver/readiness,
generic job normalization, process isolation, memory policy, storage, and the
capability-driven UI contract.

Real-model GPU validation is still intentionally separate. Backend registration
or browser readiness must not be described as a successful hardware smoke test.
