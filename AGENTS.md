# DuckMotion Agent Guide

Use this file as the first-stop guide for implementation work in DuckMotion.

## Project Snapshot

DuckMotion is a separately managed WebbDuck web plugin for local video
generation. Its runtime architecture is model-driven rather than Wan-first.
The user selects a model; DuckMotion discovers capabilities/defaults/constraints,
resolves an installed backend, and when needed resolves trusted checkpoint
provenance, an execution profile, and support assets automatically.

Current runtime families:

- Wan 2.2 through an isolated Diffusers worker.
- LTX-2.5 through an isolated two-stage Diffusers worker with synchronized audio.
- LTX-2.5 INT8 ConvRot through a dedicated runtime plus recipe-driven execution profiles.

Before major model/runtime work, read:

1. `docs/MODEL_DRIVEN_VIDEO_ARCHITECTURE.md`
2. `docs/EXECUTION_RECIPES.md`
3. `README.md`
4. `model_runtime.py`
5. `model_provenance.py` / `provenance_providers.py`
6. `model_recipes.py`
7. `model_asset_providers.py`
8. `plugin_backend.py`
9. the relevant backend/worker pair
10. the focused tests for that subsystem

## Non-Negotiable Product Rule

The selected model is the user-facing generation choice. Architecture,
backend/runtime, checkpoint-format, provenance-provider, and execution-profile
identifiers are internal routing metadata.

Do not add required Wan/LTX/architecture/backend/profile/provenance tabs or
dropdowns. Inputs and controls must be derived from selected-model capabilities,
defaults, constraints, and resolved runtime/profile semantics.

Normal setup must also remain backend-agnostic. A user should not need to know
which runtime, worker, Comfy node category, provenance service, or support-asset
folder a checkpoint uses.

## Current Repo Map

- `plugin.json`: WebbDuck web-plugin manifest.
- `plugin_backend.py`: architecture-neutral API composition root.
- `model_runtime.py`: video checkpoint descriptors, capabilities, constraints, backend resolver.
- `model_discovery.py`: local + Hugging Face cache model discovery.
- `model_provenance.py`: strong checkpoint fingerprinting, provenance registry/cache.
- `provenance_providers.py`: built-in provenance-provider composition.
- `civitai_provenance.py`: SHA256 Civitai provenance provider; setup-time only.
- `model_recipes.py`: execution-profile contracts and registry.
- `model_asset_providers.py`: normalized support-asset provider registry.
- `job_runtime.py`: generic job lifecycle and backend invocation.
- `host_runtime.py`: architecture-neutral WebbDuck runtime/GPU lease bridge.
- `runtime_services.py`: storage + host-runtime composition.
- `runtime_surfaces.py`: generic health/config/status contracts.
- `storage_runtime.py`: config/job/staging/gallery persistence.
- `storage_api.py`: generic job/staging/gallery routes.
- `wan_backend.py` / `wan_worker.py`: isolated Wan adapter/runtime.
- `ltx_backend.py` / `ltx_worker.py`: isolated standard LTX-2.5 adapter/runtime.
- `ltx_convrot_backend.py`: ConvRot format backend; resolves execution profiles.
- `ltx_convrot_worker.py`: current `ltx25_convrot_two_stage_av` worker.
- `tools/prepare_model_assets.py`: generic provenance + support-asset setup.
- `runtime_requirements/`: model-runtime-specific Python environments.
- `ui/`: capability-driven browser UI.
- `tests/`: focused contract/runtime tests.

The former monolithic `backend.py` is intentionally deleted. Do not recreate a
new architecture-specific application module under another name.

## Runtime / Setup Flow

1. WebbDuck imports `plugin_backend.py` from the plugin manifest.
2. Config stores a selected model source plus generic model/output roots.
3. Discovery produces checkpoint descriptors without loading runtime engines.
4. During setup, a format-specific asset provider may discover a checkpoint that
   needs a recipe but has no local companion.
5. Generic provenance resolution fingerprints that exact file and may query
   trusted registered providers; exactly one match may be cached.
6. Cached provenance JSON is declarative evidence only. A recipe adapter must
   still prove an installed `ExecutionProfile`; provenance names/IDs do not select it.
7. The profile supplies required assets/runtime surface/defaults and a worker entrypoint.
8. Generic asset setup resolves/downloads trusted missing support assets.
9. At generation time, `VideoJobCoordinator` persists the job and acquires
   WebbDuck's shared GPU lease.
10. `VideoBackendResolver` chooses an installed backend from the descriptor.
11. The backend reuses locally cached recipe/assets and launches the isolated worker.
12. The worker loads its model-specific pipeline, generates artifacts, and exits.
13. Generic storage normalizes/persists gallery/job metadata.
14. The GPU lease is released in coordinator cleanup.

Doctor/runtime readiness must not perform provenance network calls or recompute a
large checkpoint hash. They consume local cached provenance only.

## Ownership Boundaries

- **model discovery/introspection**: candidates -> checkpoint descriptors;
- **checkpoint provenance providers**: strong fingerprint -> canonical source + cached declarative sidecars;
- **backend resolver**: descriptor -> installed backend;
- **execution profile registry**: compatible recipe evidence -> profile;
- **asset providers**: checkpoint/recipe -> normalized asset manifest + resolved paths;
- **backend adapters**: request serialization, readiness, child process lifecycle;
- **workers**: runtime-specific imports/loading/memory/generation;
- **host runtime**: WebbDuck runtime profile and GPU lease only;
- **job coordinator**: generic job lifecycle and lease ownership;
- **storage**: config/jobs/staging/gallery/filesystem behavior;
- **UI**: selected-model capabilities and constraints only.

Generic modules must not import Wan/LTX Diffusers pipeline classes or branch on
community/vendor model names.

## Architecture Rules

- Keep model selection as the main configuration key.
- Do not require architecture, format, runtime, provenance, or recipe knowledge from API clients.
- Do not spread family-name or model-brand checks through router/job/UI/setup code.
- Unknown models fail with detection/readiness diagnostics rather than being guessed.
- Source media required/optional/unsupported is a model capability.
- Backend-specific Python dependencies belong in isolated runtime environments.
- Backend workers may use different Diffusers/Transformers versions without changing WebbDuck's host environment.
- Preserve explicit seed `0`.
- **Checkpoint-format detection must not select recipe-specific defaults.**
- **Provenance must be grounded in strong file identity (SHA256 or an equivalent provider contract), not display names.**
- **Provenance provider names, model names, version names, IDs, URLs, tags, or descriptions must never select an execution profile.**
- **Provenance may supply declarative recipe candidates; recipe adapters remain authoritative.**
- **If zero or multiple provenance providers match, refuse to guess.**
- **Recipe-specific worker/defaults/constraints/runtime nodes/assets belong to `ExecutionProfile`, not the model brand.**
- **If zero or multiple execution profiles/recipe candidates match structural evidence, refuse to guess.**
- **Support-asset setup must go through registered providers; do not add brand-specific setup commands.**
- **Profile asset defaults may provide trusted sources only for the same standard asset identity; never substitute a custom differently named asset.**
- A brand token may exist only as a weak compatibility hint for format detection when structural metadata is unavailable. It must never select provenance, a profile, or an asset table.
- Backend resource cleanup must remain resolver-driven, not family-driven.
- Public capabilities describe what the currently installed implementation can run, not merely everything an upstream checkpoint might support in theory.

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

## Standard LTX-2.5 Runtime

The standard Diffusers path is the reference distilled two-stage flow:

1. stage-1 diffusion at half final resolution;
2. latent 2x spatial upsample;
3. full-resolution stage-2 refinement using the stage-2 distilled sigma schedule;
4. synchronized audio/video encoding.

Final dimensions are multiples of 64 and frame count follows `8k+1`. The
checkpoint/runtime's explicit sigma schedules are model semantics and the public
constraint marks the sampling schedule locked. Do not replace those schedules
with a generic arbitrary-step workflow unless upstream model behavior explicitly
changes.

LTX currently tracks Diffusers main in its isolated environment. Do not upgrade
WebbDuck globally for LTX.

## LTX-2.5 ConvRot Runtime

ConvRot is a checkpoint/source format, not a model identity or a sampling recipe.
`ltx_convrot_backend.py` owns format-specific runtime readiness and child-process
lifecycle, then resolves an `ExecutionProfile`.

The current profile is `ltx25_convrot_two_stage_av`. Its implementation was
audited from a REDGraft workflow, making REDGraft a compatibility/test fixture
rather than the backend identity.

A future unrelated ConvRot checkpoint may reuse the same profile or register a
new profile. Adding a new profile should not require a new backend or UI mode.

Companion recipes are declarative only. Supported adapters may consume a native
`duckmotion_recipe` manifest or structural evidence from an exported workflow,
but DuckMotion never executes arbitrary companion workflow JSON.

If a companion is absent locally, setup may recover public JSON sidecars through
strong hash provenance. The current Civitai provider uses the public model-version
by-SHA256 endpoint, verifies that the returned version echoes the exact SHA256,
and caches only small JSON candidates. It never re-downloads the selected model.

## WebbDuck Integration Rules

DuckMotion is optional; WebbDuck must remain usable if it is absent or broken.
DuckMotion may reuse WebbDuck runtime-profile resolution, output/media paths,
and the shared GPU lease. It must not depend on WebbDuck image pipeline classes.

A WebbDuck -> DuckMotion handoff sends source media and generation context. It
does not dictate `engine=wan`, `engine=ltx`, a provenance provider, or an
execution profile.

## Verification

Run the narrowest relevant tests first, then broaden for shared contracts. Useful
sets include:

```bash
pytest -v \
  tests/test_model_runtime.py \
  tests/test_model_discovery.py \
  tests/test_model_provenance.py \
  tests/test_civitai_provenance.py \
  tests/test_provenance_recipe_integration.py \
  tests/test_runtime_readiness.py \
  tests/test_job_runtime.py \
  tests/test_wan_backend.py \
  tests/test_ltx_backend.py \
  tests/test_ltx_convrot.py \
  tests/test_recipe_driven_assets.py \
  tests/test_profile_asset_defaults.py \
  tests/test_prepare_model_runtimes.py \
  tests/test_simple_bootstrap.py \
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
