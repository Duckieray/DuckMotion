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
    +--> isolated LTX-2.5 Diffusers worker
    `--> isolated LTX-2.5 ConvRot worker
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
- `DUCKMOTION_LTX_PYTHON`;
- `DUCKMOTION_LTX_CONVROT_PYTHON`.

The ConvRot environment also owns a pinned Comfy core checkout. DuckMotion imports
that checkout as a Python library only; it does not launch a Comfy server, expose
Comfy's UI, or submit arbitrary workflow JSON over HTTP.

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

The standard isolated worker follows the distilled two-stage path:

1. half-resolution stage-1 diffusion;
2. 2x latent spatial upsampling;
3. full-resolution stage-2 refinement;
4. synchronized audio/video encoding.

Descriptor semantics therefore describe final output:

- default 1536x1024 for the standard Diffusers runtime;
- 121 frames;
- 24 fps;
- final dimensions divisible by 64;
- frame count `8k+1`;
- `generation_stages = 2`;
- `sampling_schedule_locked = true`.

The browser hides arbitrary step/guidance editing for this locked schedule.

### REDGraft LTX-2.5 INT8 ConvRot

ConvRot single-file checkpoints are detected privately and routed to
`ltx25_convrot`; the public model payload remains architecture/backend-free.
Readiness is strict: the selected checkpoint must have a companion JSON recipe
and all four declared support weights must resolve before generation starts:

- Gemma 4 12B LTX-2.5 ConvRot text encoder;
- LTX 2.3 x2 spatial latent upscaler;
- LTX 2.5 video VAE;
- LTX 2.5 audio VAE.

The current REDGraft contract is intentionally fixed rather than exposing
arbitrary Comfy sampling controls:

1. final defaults are 1152x768, 241 frames, 24 fps, CFG 1;
2. stage one runs at half final width/height;
3. I2V sources are resized to a 1536-pixel longer edge with Lanczos and passed
   through `LTXVPreprocess(img_compression=18)`;
4. positive text conditioning is paired with `ConditioningZeroOut(positive)`;
5. video and audio latents are concatenated and sampled together with Euler and
   `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`;
6. after stage one, `LTXVCropGuides` prepares the stage-two conditioning/latent;
7. video latent refinement is `LTXVLatentUpsampler` followed by bicubic
   `LatentUpscaleBy(scale_by=0.5)`; I2V then reapplies the source guide at
   strength 1.0 (stage-one guide strength is 0.7);
8. video/audio latents are recombined and sampled with Euler and
   `0.85, 0.7250, 0.4219, 0.0`;
9. final video and audio latents are separated, decoded, and muxed to one video.

Both VAEs are loaded through Comfy core `VAELoader`. The audio VAE is supplied to
`LTXVEmptyLatentAudio`, matching the pinned Comfy node contract.

The companion JSON is treated as declarative asset/recipe evidence only. It is
never executed as an arbitrary graph. See `docs/LTX25_CONVROT.md` for model
layout, environment, and validation details.

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
generic job normalization, process isolation, memory policy, storage, the
capability-driven UI contract, and the fixed REDGraft ConvRot recipe primitives.

Real-model GPU validation is still intentionally separate. Backend registration,
asset readiness, or passing unit/contract tests must not be described as a
successful hardware smoke test.
