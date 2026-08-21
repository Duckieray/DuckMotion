# Model-Driven Video Architecture

Status: **implementation design / migration target**

This document defines the target architecture for DuckMotion as a model-driven video generation workspace. It complements WebbDuck's `docs/ARCHITECTURE_AGNOSTIC_GENERATION.md` and applies the same product rule to video:

> The user selects a video model. DuckMotion determines architecture, runtime backend, supported inputs/outputs, defaults, and constraints automatically.

The current DuckMotion implementation is Wan2.2 image-to-video specific. This plan does not claim that the refactor has already landed.

## Product Contract

Normal DuckMotion generation should not require the user to choose:

- Wan vs LTX vs another video architecture;
- Diffusers vs an official/reference runtime;
- GGUF/FP8/BF16 loader implementation;
- in-process vs subprocess execution;
- pipeline class names.

The user-facing workflow is:

```text
Select video model
      |
      v
DuckMotion loads model profile/capabilities
      |
      +--> source image required/optional/unavailable?
      +--> text-only generation supported?
      +--> audio output supported?
      +--> video/reference inputs supported?
      +--> valid duration/FPS/resolution/defaults?
      |
      v
Configure supported controls
      |
      v
Queue generation
      |
      v
Automatic backend/runtime resolution
      |
      v
Video result (+ audio/other artifacts when supported)
```

Architecture names may appear in a model-details/debug panel, but they are not primary navigation or required configuration.

## Core Principles

1. **The selected model is the routing key.** There is no required engine selector.
2. **Capabilities define the Create workspace.** Input staging and controls adapt to the selected model.
3. **Architecture and runtime are separate.** One architecture may have multiple backend implementations.
4. **Backend implementation is replaceable.** Moving a model from Diffusers to an official runtime or optimized loader must not change the public job contract.
5. **Subprocess isolation is a feature, not a workaround.** Video models often need dependency stacks too large or too volatile to share safely with WebbDuck.
6. **GPU lease coordination remains mandatory.** Isolated child processes still run under the parent-owned WebbDuck GPU lease.
7. **Output is multimodal.** The normalized result contract must be able to represent video, audio, poster frames, frame sequences, metadata, and future artifacts.
8. **Legacy Wan behavior remains compatible.** The first refactor step should wrap the current implementation, not replace it.
9. **Unknown models are not guessed.** Unsupported assets should expose detection/readiness diagnostics.
10. **WebbDuck handoff sends media/context, not an engine choice.** DuckMotion decides whether the currently selected video model can consume the input.

## Audit: Current DuckMotion Coupling

The current repo has useful runtime machinery, but nearly every model-facing path assumes Wan2.2.

### Reusable seams

The following are strong foundations for the refactor:

- plugin-local persistent job queue and job metadata;
- shared WebbDuck runtime profile reuse;
- shared GPU lease acquisition/release;
- pre-generation cleanup of WebbDuck models;
- isolated child-process generation path with progress-file communication;
- timeout/error normalization and persisted child crash logs;
- configurable memory policy and CUDA/offload behavior;
- local output/gallery storage;
- WebbDuck recent-image discovery/staging flow.

These should become generic video-runtime services rather than being rewritten.

### Wan-specific assumptions to remove from top-level orchestration

#### Product identity and UI

`plugin.json`, the README, and the UI identify DuckMotion as a "Wan2.2 I2V" workspace. The Create view requires staging an image and labels generation as Wan2.2 I2V.

The target UI should identify the selected **video model**, then adapt source requirements and controls from capabilities.

#### Configuration

Current configuration centers on one `model_id_or_path` plus Wan-oriented defaults. That single selected-model concept is good and should remain, but configuration should resolve a model descriptor rather than assume the path points to Wan.

`runtime_backend` currently distinguishes internal implementation choices such as standard Diffusers vs Diffusers+GGUF while all choices are still Wan. That distinction is useful, but should be moved behind a model/backend resolver and not exposed as a model-family choice.

#### Discovery

Discovery currently scans Wan-specific locations such as `checkpoint/wan` and contains Wan-specific directory and GGUF identification functions.

New preferred neutral location:

```text
<models-root>/
  checkpoints/
    video/
      <any nested organization>
```

Legacy `checkpoint/wan` / `checkpoints/wan` roots must remain supported without forcing users to move existing models.

#### Request model

`GeneratePayload.image_path` is currently required. That hard-codes image-to-video as the only operation.

The generic job contract needs optional source slots whose requirement is determined by the selected model:

- source image;
- source video;
- source audio;
- multiple reference images/media where supported.

#### Runtime probe and loader

Current readiness probes import `WanImageToVideoPipeline`, and `_get_or_load_pipeline()` ultimately builds a Wan pipeline. GGUF selection is also Wan-specific.

These are backend-adapter responsibilities.

#### Generation

`_generate_frames_with_diffusers()` assumes:

- an input image exists;
- the model returns video frames;
- `num_inference_steps` maps directly to progress;
- guidance/negative prompt semantics are Wan-compatible.

LTX-2.5 demonstrates why this cannot be the universal contract: its official distilled Diffusers recipe uses explicit sigma schedules, can generate synchronized audio, and supports text/image/video/audio-conditioned workflows.

#### Output writer

`_write_video_outputs()` currently normalizes frames, writes a poster, and exports an MP4. The result model must expand to support audio and already-muxed outputs without forcing every backend through frame-only export.

## Target Domain Model

### `VideoModelDescriptor`

Every selectable model should resolve to a normalized descriptor.

Illustrative schema:

```python
@dataclass(frozen=True)
class VideoModelDescriptor:
    id: str
    name: str
    path_or_repo: str
    source: str
    media_type: str             # video
    artifact_format: str        # diffusers, component_pack, single_file, gguf_pack, ...

    architecture: str | None    # internal resolver key
    variant: str | None         # i2v, distilled, full, etc. when detectable
    detection_confidence: str
    diagnostics: tuple[str, ...]

    capabilities: VideoCapabilities
    defaults: dict
    constraints: dict
    runtime_hints: dict
```

The architecture field is intentionally internal. `variant` must not be inferred only from filenames when authoritative configuration is available.

### Capability model

Video capabilities need to express both accepted conditioning and produced artifacts.

Example:

```json
{
  "operations": {
    "text_to_video": true,
    "image_to_video": true,
    "video_to_video": false,
    "audio_to_video": false,
    "multi_shot": false
  },
  "inputs": {
    "image": "optional",
    "video": "unsupported",
    "audio": "unsupported",
    "multi_reference": false
  },
  "outputs": {
    "video": true,
    "audio": true,
    "muxed_audio_video": true
  },
  "controls": {
    "negative_prompt": true,
    "guidance": true,
    "steps": true,
    "fps": true,
    "duration": true
  }
}
```

Input states should support at least:

- `required`;
- `optional`;
- `unsupported`.

### Constraints

Examples include:

- width/height multiples and allowed aspect/resolution ranges;
- frame-count alignment requirements;
- min/max duration;
- supported frame rates;
- maximum reference-media counts;
- whether duration is explicit or model-predicted;
- whether a distilled model expects a fixed sigma schedule instead of arbitrary steps;
- whether audio can be disabled;
- runtime memory/preflight constraints.

### Defaults

Effective settings should follow the same precedence as WebbDuck:

1. explicit user choice;
2. checkpoint-specific saved defaults;
3. exact model-variant/backend recommendations;
4. DuckMotion generic fallback.

Do not force a generic `steps=30, guidance=5` worldview onto models whose official inference recipe uses different semantics.

## Model Discovery and Introspection

### Discovery roots

Scan neutral video roots first for new installs while retaining all current Wan locations:

```text
<models-root>/checkpoints/video
<models-root>/checkpoint/video
<models-root>/checkpoints/wan      # legacy
<models-root>/checkpoint/wan       # legacy
configured DuckMotion models dir
Hugging Face cache
```

The root is only a discovery hint. The folder name must not determine architecture.

### Detector chain

Suggested evidence order:

1. explicit DuckMotion model override/manifest;
2. Diffusers `model_index.json` pipeline/component classes;
3. component configs;
4. official component-pack filenames + config metadata;
5. safetensors metadata/tensor signatures;
6. GGUF metadata/tensor signatures;
7. filename/path heuristics as low-confidence fallback.

Architecture-specific scanners such as today's `_is_wan_diffusers_model_dir()` and Wan GGUF pairing should become registered detectors/format handlers.

### Model readiness

Discovery and readiness are different:

- **discovered**: DuckMotion can identify the model and show it in the model list;
- **ready**: at least one compatible backend is installed and runtime-compatible;
- **unsupported**: identified but no adapter exists;
- **unknown**: insufficient evidence to identify safely.

The UI should show readiness without asking the user to select the backend manually.

## Backend Resolution

### Backend interface

```python
class VideoGenerationBackend(Protocol):
    backend_id: str

    def accepts(self, model: VideoModelDescriptor) -> bool: ...
    def readiness(self, model, runtime_profile) -> BackendReadiness: ...
    def validate(self, model, request) -> None: ...
    def generate(self, model, request, runtime, progress, cancel_event) -> VideoGenerationResult: ...
    def unload(self) -> None: ...
```

Backends own their pipeline class, dependency versions, quantization support, placement strategy, and output decoding.

### Resolver

The resolver considers:

- selected model descriptor;
- installed backend adapters;
- runtime profile/hardware;
- local optimized model artifacts;
- dependency/environment availability;
- explicit advanced override only if one exists for debugging.

A user selecting an LTX model should not then have to choose "LTX backend." A user selecting a Wan GGUF pack should not need to understand that a hybrid Diffusers+GGUF loader was chosen.

### Initial adapter families

The implementation should begin by extracting existing Wan behavior into adapters such as:

- `WanDiffusersBackend`;
- `WanDiffusersGgufBackend` (or an internal format strategy owned by the Wan backend).

Then add LTX implementations, potentially including:

- `LtxOfficialPipelinesBackend`;
- `LtxDiffusersBackend`.

Those names are implementation details. The resolver may prefer one and fall back to another based on readiness and model packaging.

## Process Isolation

DuckMotion already has a strong subprocess pattern. Generalize it instead of duplicating it per architecture.

Suggested boundary:

```text
parent plugin process
  |
  | owns job record + GPU lease
  | chooses backend execution spec
  v
isolated runtime launcher
  |
  +--> environment / interpreter
  +--> backend ID
  +--> normalized model descriptor
  +--> normalized generation request
  +--> progress channel
  +--> result path
  v
child runtime
```

The child should run a small generic entrypoint that dispatches to the resolved backend. Do not make the child entrypoint itself a giant Wan/LTX conditional; use the backend registry there too.

### Isolation requirements

Every isolated backend must support:

- parent-owned GPU lease and heartbeat;
- structured progress updates;
- normalized success/error result payloads;
- load vs generation timeout policies;
- cancellation/termination semantics;
- child log capture;
- deterministic cleanup of temporary payload/result files;
- explicit environment/interpreter selection.

This is particularly important for LTX-2.5 because the official `ltx-pipelines` path currently recommends a newer Python/CUDA/PyTorch stack than WebbDuck's shared environment, while its Diffusers support is also newer than the stable Diffusers version currently pinned by WebbDuck. DuckMotion should be able to use either path without forcing a global environment upgrade.

## Generic Generation Request

Replace the I2V-only payload with a model-neutral request while preserving `image_path` as a compatibility alias during migration.

Illustrative shape:

```python
class VideoGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None

    source_image: str | None = None
    source_video: str | None = None
    source_audio: str | None = None
    references: list[str] = []

    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    num_frames: int | None = None
    fps: float | None = None
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = None

    advanced: dict = {}
```

Server-side normalization determines operation intent from selected model capabilities + supplied sources.

Examples:

- no source + text-to-video capable model -> text-to-video;
- source image + image-to-video capable model -> image-to-video;
- source image when selected model requires image -> valid;
- no source when selected model requires image -> validation error before model load;
- source image with a text-only model -> unsupported-input validation error.

## Result and Artifact Model

The backend should return normalized artifacts instead of only a frame list.

```python
@dataclass
class VideoArtifact:
    kind: str        # video, audio, poster, frames, metadata
    path: str
    mime_type: str | None = None

@dataclass
class VideoGenerationResult:
    artifacts: list[VideoArtifact]
    resolved_seed: int | None
    effective_settings: dict
    model_snapshot: dict
    backend_snapshot: dict
    warnings: list[str]
    performance: dict
```

The output writer becomes an artifact finalizer:

- if a backend returns frames only, DuckMotion can encode them;
- if a backend returns video + audio arrays, DuckMotion can mux them;
- if a backend already writes a muxed MP4, DuckMotion should not decode/re-encode just to satisfy an old frame-oriented path;
- poster extraction should be generic and optional;
- gallery metadata should list all artifacts for the run.

## LTX-2.5 Integration Target

LTX-2.5 is the first non-Wan architecture that should validate this design.

As verified against the official model documentation on 2026-08-20:

- LTX-2.5 can generate synchronized video and audio;
- official supported conditioning includes text, image, video, and audio workflows;
- native multishot generation is supported;
- an official split-component `ltx-pipelines` path exists;
- a Diffusers-compatible model pack exists as `Lightricks/LTX-2.5-Diffusers`;
- the current Diffusers support requires a newer/mainline Diffusers implementation rather than assuming DuckMotion's existing stable Wan dependency set;
- distilled inference has model-specific scheduling/guidance semantics that should be represented by model defaults/constraints instead of forced through Wan's generic step/guidance assumptions.

### LTX-specific capabilities to model

Initial implementation can stage capability rollout instead of attempting every official workflow at once.

Recommended order:

1. text-to-video + synchronized audio;
2. image-to-video + synchronized audio;
3. model-driven duration/FPS controls;
4. two-stage/upscale path when required by the selected variant;
5. multishot;
6. video/audio conditioning workflows.

The UI should expose a capability only when the selected model/backend combination is ready.

### LTX output

A successful LTX job may return:

- muxed MP4;
- video stream/frames;
- audio stream/waveform;
- poster;
- metadata.

The gallery should treat the muxed video as the primary artifact when available and retain separate audio only when useful for inspection/reuse.

## Wan Migration Target

Do not rewrite the Wan loader first.

Move current behavior behind a model/backend boundary in stages:

1. create a Wan descriptor from today's discovery data;
2. wrap `_get_or_load_pipeline()` and `_generate_frames_with_diffusers()` behind a `Wan...Backend` without functional changes;
3. preserve current GGUF/hybrid selection as an internal backend/runtime strategy;
4. make `image` a `required` input capability for the current Wan I2V model;
5. keep current memory policy/safety checks, but make them backend-owned hooks;
6. update UI labels from "Wan Model" / "Generate Wan2.2 I2V" to model-neutral labels only after descriptor-driven behavior is in place.

This gives LTX a clean adjacent path instead of adding another branch to Wan functions.

## UI Migration

There should still be one selected video model.

### Setup view

Target labels:

- `Video Model` rather than `Wan Model`;
- `Discovered Models` rather than `Discovered local Wan models`;
- neutral model root/cache help text.

The readiness panel should report:

- selected model readiness;
- compatible backend installed/unavailable;
- runtime compatibility;
- required model components missing;
- estimated memory/preflight warnings.

Architecture/backend details can be hidden in an expandable diagnostic section.

### Create view

The Create view should dynamically configure itself from capabilities.

Examples:

- Wan I2V selected: staging/source-image panels remain required.
- LTX text-to-video selected with no source: staging is optional and generation remains enabled.
- LTX image-to-video: user may choose a staged/WebbDuck image.
- Audio-capable model: show audio-generation/output controls where meaningful.
- Model with no negative-prompt semantics: hide/disable that field.
- Model with a fixed distilled schedule: do not present arbitrary steps as though they are meaningful.

No architecture tabs are required.

## WebbDuck -> DuckMotion Handoff

The handoff contract is media-centric.

Recommended request:

```json
{
  "source_image": "outputs/<run>/0.png",
  "prompt": "optional inherited prompt",
  "source_metadata": {
    "model": "image-model-name",
    "seed": 1234,
    "run": "..."
  }
}
```

DuckMotion then checks the **currently selected video model**:

- if image-to-video is supported, accept and stage/use the image;
- if an image is required, the handoff satisfies the requirement;
- if image input is unsupported, explain that the selected video model cannot consume the source.

WebbDuck should not tell DuckMotion `engine=ltx` or `engine=wan` during a normal handoff.

Future handoffs can carry multiple references or source video/audio without changing this principle.

## API Evolution

Keep existing routes where practical, but return model descriptors/readiness rather than Wan-only detection results.

Suggested endpoints/contracts:

- `GET /models/discover` -> all identified video models + readiness summaries;
- `GET /models/{id}/profile` -> capabilities, constraints, defaults, diagnostics;
- `GET /health` -> selected-model readiness + generic backend/runtime readiness;
- `POST /config` -> selected model plus user-level defaults, not a required architecture;
- `POST /generate` -> normalized generic request;
- legacy `image_path` accepted as alias for `source_image` during migration.

Job records should snapshot:

- selected model descriptor;
- chosen backend ID and execution mode for diagnostics;
- normalized operation intent;
- effective settings;
- all output artifacts.

The chosen backend remains diagnostic metadata, not a required client input.

## Migration Plan

### Phase 0 - Regression coverage

- Lock current Wan Diffusers and GGUF planning behavior with tests.
- Cover local/HF discovery, model readiness, memory policy, safety preflight, image staging, isolated subprocess success/failure, output metadata, and GPU lease release.
- Do not change dependencies yet.

### Phase 1 - Video descriptors and neutral discovery

- Add candidate/descriptor/capability/constraint models.
- Convert current Wan discovery results into descriptors.
- Add neutral video roots while preserving legacy Wan roots.
- Enrich model discovery/readiness APIs without changing generation behavior.

Exit criterion: the current Wan model is represented as a descriptor and existing UI still works.

### Phase 2 - Backend registry and Wan wrapper

- Add video backend protocol/registry/resolver.
- Move current Wan loading/generation behind the adapter boundary.
- Generalize the pipeline cache key/session ownership.
- Make memory policy and safety preflight backend hooks.
- Generalize subprocess launcher to dispatch a selected backend.

Exit criterion: existing Wan generation runs through a generic model-selected backend path.

### Phase 3 - Capability-driven request/UI/output

- Make source image optional at the API schema level and required by selected-model validation when appropriate.
- Add generic operation resolution.
- Make Create controls/staging capability-aware.
- Replace frame-only result assumptions with the artifact result model.
- Keep current gallery compatible with video-only jobs while adding audio artifact metadata.

Exit criterion: the UI can represent a text-to-video-capable model without adding an architecture tab.

### Phase 4 - LTX-2.5 backend

- Add LTX-2.5 discovery/detection fixtures.
- Implement an isolated official `ltx-pipelines` backend and/or isolated Diffusers backend based on runtime readiness.
- Support text-to-video with synchronized audio first.
- Add image-to-video using the same selected-model workflow.
- Persist/mux audio artifacts correctly.

Exit criterion: selecting an LTX model automatically changes DuckMotion from required-I2V staging to the capabilities LTX exposes, with no engine selector.

### Phase 5 - Advanced LTX workflows

- Add two-stage/upscale requirements where selected variant needs them.
- Add multishot capabilities/UI.
- Add video/audio conditioning workflows as the backend support stabilizes.

### Phase 6 - Additional architectures/formats

- Add future video architectures by detector + backend adapter + capability tests.
- Add quantized formats as format/runtime strategies, not new user-facing engines.

## Proposed File Boundaries

Names are illustrative.

```text
duckmotion/
  models/
    descriptors.py
    discovery.py
    introspection/
  backends/
    base.py
    registry.py
    wan.py
    ltx.py
  runtime/
    isolation.py
    memory.py
    session.py
  jobs/
    models.py
    worker.py
  artifacts.py
```

The current single-file `backend.py` is large enough that the architecture refactor is also an opportunity to split ownership. Do it incrementally while preserving the plugin's `get_router()` entrypoint expected by WebbDuck.

A reasonable first extraction is pure model discovery/descriptors, followed by backend adapters, then job/output/runtime helpers.

## Implementation Rules for Coding Agents

- Do not add Wan/LTX architecture tabs or a required engine dropdown.
- Do not add `if selected_engine == "ltx"` branches throughout UI/job/server code.
- Do not require `image_path` universally after generic request support lands.
- Do not make every model expose `steps`, `guidance`, negative prompt, or the same duration semantics.
- Do not force LTX dependencies into WebbDuck's stable environment if an isolated runtime is safer.
- Do not remove current Wan GGUF/memory/safety behavior while extracting it.
- Do not assume every backend returns a list of PIL frames.
- Do preserve plugin API compatibility during migration.
- Do preserve parent-owned GPU lease coordination for child processes.
- Do snapshot chosen backend/execution details in job metadata for debugging even though they are hidden from the normal workflow.

## Testing Matrix

| Area | Required tests |
| --- | --- |
| Discovery | neutral roots, legacy Wan roots, HF cache, component packs, unknown assets |
| Descriptor | Wan and LTX capabilities/defaults/constraints, serialization, diagnostics |
| Resolver | backend priority/readiness/fallback, missing dependency behavior |
| Request | required/optional/unsupported source media validation |
| Wan regression | existing I2V Diffusers, GGUF planning, memory policy, safety checks |
| Isolation | progress, heartbeat, timeout, crash logs, cleanup, cancellation |
| LTX | text-to-video+audio, image-to-video+audio, model-specific scheduling/defaults |
| Output | video-only, muxed AV, separate audio, poster, metadata artifacts |
| UI | one model selector, capability-driven staging/controls, no engine selector |
| Cross-app | WebbDuck image handoff accepted/rejected from selected-model capability |
| Switching | Wan -> LTX -> Wan resource cleanup and no stale runtime state |

## Definition of Done

The model-driven DuckMotion migration is complete when:

- the user selects a video model rather than an engine;
- current Wan2.2 I2V behavior remains regression-covered;
- LTX-2.5 can run from the same model-selection/job workflow;
- source-image requirements come from model capabilities rather than the global request schema;
- synchronized audio results can be represented and stored without abusing a frame-only output path;
- backend/runtime selection is automatic and may use isolated environments;
- switching between model architectures safely releases resources and keeps GPU lease ownership correct;
- discovery supports neutral video-model roots plus legacy Wan roots;
- unsupported/unknown models return actionable readiness diagnostics;
- top-level plugin UI, routing, job management, and artifact storage do not contain architecture-specific routing logic outside designated discovery/resolver/backend modules.
