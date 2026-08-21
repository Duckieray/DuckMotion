# DuckMotion Agent Guide

Use this file as the first-stop guide for implementation work in DuckMotion.

## Project Snapshot

DuckMotion is a separately managed WebbDuck web plugin. The current implementation is a Wan2.2 image-to-video workspace using a local Diffusers runtime, plugin-local job queue, shared WebbDuck runtime profile, shared GPU lease, optional subprocess isolation, and plugin-local video gallery.

The **target architecture is model-driven rather than Wan-first**. Before adding another video architecture or making major model/runtime changes, read:

1. `docs/MODEL_DRIVEN_VIDEO_ARCHITECTURE.md`
2. `README.md`
3. `backend.py`
4. `ui/index.html`
5. `ui/app.js`
6. `tests/test_backend_memory_policy.py`

## Non-Negotiable Product Rule

The user selects a video model. DuckMotion automatically detects its architecture, capabilities, backend/runtime implementation, defaults, and constraints.

Do not solve new model support by adding required `Wan`, `LTX`, or other architecture tabs/dropdowns.

Architecture/backend details may be shown in diagnostics, but they are not required inputs for normal generation.

## Current Repo Map

- `plugin.json`: WebbDuck web-plugin manifest.
- `backend.py`: current router, config, discovery, queue, Wan runtime, isolation, memory/safety logic, output writing.
- `ui/index.html`: Setup/Create/Jobs/Gallery markup.
- `ui/app.js`: browser-side API/state/rendering behavior.
- `ui/styles.css`: plugin styles.
- `tools/install_webbduck_plugin.py`: installs/copies the plugin into WebbDuck plugin roots.
- `tests/test_backend_memory_policy.py`: focused backend/memory/runtime tests.
- `docs/MODEL_DRIVEN_VIDEO_ARCHITECTURE.md`: target architecture and migration plan.

## Current Runtime Flow

1. WebbDuck discovers `plugin.json` and imports `backend.py`.
2. `get_router()` starts the DuckMotion worker and returns the plugin API router.
3. Setup/config selects one model source plus runtime/output defaults.
4. A generation request becomes a persisted DuckMotion job.
5. The DuckMotion worker acquires the shared WebbDuck GPU lease.
6. DuckMotion asks WebbDuck to release generation/captioning VRAM where possible.
7. Generation executes in-process or through the existing isolated child-process path.
8. The current implementation loads a Wan Diffusers pipeline, generates frames, and exports a video.
9. Job/output metadata is persisted under DuckMotion plugin state/output locations.
10. The GPU lease is released in `finally` cleanup.

## Target Ownership Boundaries

As the architecture refactor proceeds, prefer these responsibilities even if file extraction happens gradually:

- **model discovery/introspection**: find candidates and produce video-model descriptors;
- **backend registry/resolver**: choose a compatible installed backend from a model descriptor + runtime;
- **backend adapters**: own architecture/runtime-specific loading, validation, generation, and unloading;
- **runtime isolation**: generic child-process launcher/progress/error protocol;
- **job worker**: generic queue/job lifecycle and GPU lease ownership;
- **artifact finalization**: persist video/audio/poster/metadata without assuming frame-only results;
- **UI**: render selected-model capabilities, not architecture-specific pages.

## Architecture Refactor Rules

- Keep the selected **model** as the main configuration key.
- Do not require architecture knowledge from API clients.
- Do not spread architecture checks through router/job/UI code.
- Treat `architecture`, `artifact_format`, and `runtime backend` as separate descriptor concepts.
- Preserve current Wan Diffusers/GGUF/memory/safety behavior while extracting it behind adapters.
- Make source media required/optional/unsupported through selected-model capabilities; do not keep image input globally mandatory.
- Allow backends to return normalized artifacts instead of forcing every model to return PIL frame lists.
- Preserve shared GPU lease ownership across subprocess runtimes.
- Prefer isolated environments when a new model's Python/Diffusers/Transformers/CUDA requirements conflict with WebbDuck's stable environment.
- Unknown model assets should fail with model-detection/readiness diagnostics rather than being guessed as Wan.

## LTX-2.5 Target

LTX-2.5 is the first non-Wan validation architecture for the model-driven design.

The first implementation should prioritize:

1. text-to-video with synchronized audio;
2. image-to-video with synchronized audio;
3. model-specific duration/scheduling/default behavior;
4. output artifact/muxing support;
5. later multishot and additional conditioning inputs.

Use the official split-component/runtime path and/or Diffusers-compatible pack behind backend adapters. Do not globally upgrade WebbDuck's environment merely to make the first LTX path load.

## WebbDuck Integration Rules

DuckMotion is optional. WebbDuck must remain usable when DuckMotion is missing or broken.

DuckMotion may reuse:

- WebbDuck runtime profile resolution;
- WebbDuck path/model-root hints;
- WebbDuck output paths for image handoff;
- WebbDuck shared GPU lease;
- WebbDuck model unload hooks.

Do not make DuckMotion depend on WebbDuck image-generation internals such as SDXL pipeline classes.

A WebbDuck -> DuckMotion handoff should send source media and generation context. It should not dictate `engine=wan` or `engine=ltx`.

## Verification

For documentation-only changes, verify file presence/content and branch diff.

For runtime changes, run the narrowest relevant tests first, then broaden when shared contracts change. At minimum consider:

```bash
pytest -v tests/test_backend_memory_policy.py
```

When the model-driven refactor adds new test modules, cover:

- discovery/descriptors;
- backend resolver/readiness;
- generic request validation;
- Wan regression behavior;
- subprocess progress/error/lease handling;
- LTX video+audio generation;
- model switching/resource cleanup;
- UI capability behavior.

Never claim a real-model smoke test ran if the required weights/hardware/runtime were not available.

## Git Rules

- Work on a feature branch, not `main`.
- Keep commits scoped to the requested work.
- Do not commit model weights, outputs, plugin state, child logs, secrets, or machine-specific paths.
- Do not force-push or rewrite unrelated history.
- Update `docs/MODEL_DRIVEN_VIDEO_ARCHITECTURE.md` when implementation decisions materially change the target contract.
