# DuckMotion Agent Guide

Use this file as the first-stop guide for implementation work in DuckMotion.

## Project Snapshot

DuckMotion is a separately managed WebbDuck web plugin for local video
generation. Its runtime architecture is model-driven rather than Wan-first.
The user selects a model; DuckMotion discovers capabilities/defaults/constraints
and resolves an installed backend automatically.

Current runtime families:

- Wan 2.2 through an isolated Diffusers worker.
- LTX-2.5 through an isolated two-stage Diffusers worker with synchronized audio.

Before major model/runtime work, read:

1. `docs/MODEL_DRIVEN_VIDEO_ARCHITECTURE.md`
2. `README.md`
3. `model_runtime.py`
4. `plugin_backend.py`
5. the relevant backend/worker pair
6. the focused tests for that subsystem

## Non-Negotiable Product Rule

The selected model is the user-facing generation choice. Architecture and
backend/runtime identifiers are internal routing metadata.

Do not add required Wan/LTX/architecture/backend tabs or dropdowns. Inputs and
controls must be derived from selected-model capabilities, defaults, and
constraints.

## Current Repo Map

- `plugin.json`: WebbDuck web-plugin manifest.
- `plugin_backend.py`: architecture-neutral API composition root.
- `model_runtime.py`: video descriptors, capabilities, constraints, backend resolver.
- `model_discovery.py`: local + Hugging Face cache model discovery.
- `job_runtime.py`: generic job lifecycle and backend invocation.
- `host_runtime.py`: architecture-neutral WebbDuck runtime/GPU lease bridge.
- `runtime_services.py`: storage + host-runtime composition.
- `runtime_surfaces.py`: generic health/config/status contracts.
- `storage_runtime.py`: config/job/staging/gallery persistence.
- `storage_api.py`: generic job/staging/gallery routes.
- `wan_backend.py` / `wan_worker.py`: isolated Wan adapter/runtime.
- `ltx_backend.py` / `ltx_worker.py`: isolated LTX-2.5 adapter/runtime.
- `runtime_requirements/`: model-runtime-specific Python environments.
- `ui/`: browser UI; until the final capability-driven rewrite it may contain older presentation assumptions.
- `tests/`: focused contract/runtime tests.

The former monolithic `backend.py` is intentionally deleted. Do not recreate a
new architecture-specific application module under another name.

## Runtime Flow

1. WebbDuck imports `plugin_backend.py` from the plugin manifest.
2. Config stores a selected model source plus generic model/output roots.
3. Discovery produces model descriptors without loading runtime engines.
4. A generation request is validated against the selected descriptor.
5. `VideoJobCoordinator` persists the job and acquires WebbDuck's shared GPU lease.
6. `VideoBackendResolver` chooses an installed backend from the descriptor.
7. The backend launches its isolated worker environment.
8. The worker loads its model-specific pipeline, generates artifacts, and exits.
9. Generic storage normalizes/persists gallery/job metadata.
10. The GPU lease is released in coordinator cleanup.

## Ownership Boundaries

- **model discovery/introspection**: candidates -> descriptors;
- **backend resolver**: descriptor -> installed backend;
- **backend adapters**: request serialization, readiness, child process lifecycle;
- **workers**: model-family-specific imports/loading/memory/generation;
- **host runtime**: WebbDuck runtime profile and GPU lease only;
- **job coordinator**: generic job lifecycle and lease ownership;
- **storage**: config/jobs/staging/gallery/filesystem behavior;
- **UI**: selected-model capabilities and constraints only.

Generic modules must not import Wan/LTX Diffusers pipeline classes.

## Architecture Rules

- Keep model selection as the main configuration key.
- Do not require architecture knowledge from API clients.
- Do not spread family-name checks through router/job/UI code.
- Unknown models fail with detection/readiness diagnostics rather than being guessed.
- Source media required/optional/unsupported is a model capability.
- Backend-specific Python dependencies belong in isolated runtime environments.
- Backend workers may use different Diffusers/Transformers versions without changing WebbDuck's host environment.
- Preserve explicit seed `0`.
- Model-specific schedules/defaults belong to descriptors or workers, not UI engine presets.
- Backend resource cleanup must remain resolver-driven, not family-driven.
- Public capabilities describe what the currently installed backend can run, not merely everything an upstream checkpoint might support in theory.

## Wan Runtime

The current isolated Diffusers backend supports pure Wan T2V and pure Wan I2V
checkpoints. A pure I2V model requires a source image.

`Wan2.2-TI2V-5B` is a special case: the upstream checkpoint supports T2V and
I2V, but current Diffusers `WanPipeline` exposes only its text-conditioned path.
Until DuckMotion gains a native/unified TI2V image-conditioning runtime, its
public descriptor must expose `text_to_video=true` and `image_to_video=false`.
Internal detection metadata may record the upstream I2V capability so it can be
enabled later without rediscovering the model family. Do not offer a source-image
workflow that the selected backend cannot execute.

TI2V-5B also has checkpoint-specific published defaults: 1280x704 landscape,
121 frames, 24 fps, 50 steps, and guidance 5.0. Turbo variants may override the
step/guidance defaults while retaining model-specific size/frame defaults.

Wan memory policy belongs in `wan_worker.py`. On constrained GPUs, use Diffusers
offload APIs rather than adding runtime selectors to the UI. Very large A14B
checkpoints may still be impractical on a host despite disk/group offload; do
not claim feasibility without a real smoke run.

## LTX-2.5 Runtime

The production path is the reference distilled two-stage flow:

1. stage-1 diffusion at half final resolution;
2. latent 2x spatial upsample;
3. full-resolution stage-2 refinement using the stage-2 distilled sigma schedule;
4. synchronized audio/video encoding.

Final dimensions are multiples of 64 and frame count follows `8k+1`. The
checkpoint's explicit sigma schedules are model semantics and the public
constraint marks the sampling schedule locked. Do not replace those schedules
with a generic arbitrary-step workflow unless upstream model behavior explicitly
changes.

LTX currently tracks Diffusers main in its isolated environment. Do not upgrade
WebbDuck globally for LTX.

## WebbDuck Integration Rules

DuckMotion is optional; WebbDuck must remain usable if it is absent or broken.
DuckMotion may reuse WebbDuck runtime-profile resolution, output/media paths,
and the shared GPU lease. It must not depend on WebbDuck image pipeline classes.

A WebbDuck -> DuckMotion handoff sends source media and generation context. It
does not dictate `engine=wan` or `engine=ltx`.

## Verification

Run the narrowest relevant tests first, then broaden for shared contracts. Useful
sets include:

```bash
pytest -v \
  tests/test_model_runtime.py \
  tests/test_model_discovery.py \
  tests/test_runtime_readiness.py \
  tests/test_job_runtime.py \
  tests/test_wan_backend.py \
  tests/test_ltx_backend.py \
  tests/test_backend_memory_policy.py \
  tests/test_plugin_backend.py \
  tests/test_legacy_backend_removed.py
```

Never claim a real-model smoke test ran unless the required weights, runtime,
and hardware were actually used.

## Git Rules

- Work on a feature branch, not `main`.
- Keep commits scoped.
- Do not commit model weights, outputs, plugin state, child logs, secrets, or machine-specific paths.
- Do not force-push or rewrite unrelated history.
- Update architecture docs when implementation materially changes the contract.
