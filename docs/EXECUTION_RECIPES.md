# DuckMotion Execution Recipes

Execution recipes are DuckMotion's private extension point for checkpoints that
share an architecture/weight format but need different runtime behavior.

Users never select recipes. They select a model/checkpoint; DuckMotion resolves a
compatible recipe automatically.

## Core contract

A checkpoint descriptor answers **what the model is**:

```text
architecture
source/weight format
capabilities
checkpoint-level defaults/constraints
```

Checkpoint provenance answers **where this exact file came from**:

```text
strong content fingerprint
trusted provenance provider
canonical source/model-version identity
cached declarative recipe candidates
```

An `ExecutionProfile` answers **which trusted runtime topology can execute it**:

```text
profile id
compatible architecture + format
worker entrypoint
required support-asset roles
trusted standard asset sources
required runtime APIs/nodes
recipe defaults/constraints
structural recipe evidence
```

A validated companion may additionally answer **how this checkpoint author tuned
that topology**:

```text
sampler
stage sigma schedules
CFG
guide strengths
stage-to-stage noise policy
```

Those tuning values are not architecture identity. DuckMotion normalizes only an
allow-listed subset it understands and never executes arbitrary workflow JSON.

Display/vendor/community names are not part of runtime or recipe selection.

## Provenance resolution

A missing local recipe does not justify guessing from a checkpoint filename.
During setup, DuckMotion may identify a provider-owned checkpoint using a strong
content fingerprint and cache trusted declarative sidecars locally.

The first built-in provider is Civitai SHA256 provenance:

```text
local checkpoint
  -> SHA256
  -> public Civitai model-version by-hash lookup
  -> exact SHA256 echoed by that version
  -> cache version metadata + public JSON sidecars
```

Only small JSON sidecars are downloaded by provenance providers. The selected
checkpoint is never replaced or re-downloaded. The provenance cache lives under
`~/.cache/duckmotion/provenance/` by default and is tied to the checkpoint's
local file-change state. Once a recipe sidecar has been materialized, doctor and
runtime readiness use the local cache only; they do not contact provenance
services.

A provenance match **does not select an execution profile**. Provider model name,
version name, IDs, URLs, tags, and branding are diagnostic metadata only. Cached
JSON must still pass a recipe adapter below.

If zero provenance providers match, setup reports that fact and leaves the model
blocked. If multiple providers claim the same checkpoint, DuckMotion refuses to
choose one automatically.

## Optional provider credentials

Public provider access is anonymous by default. A normal DuckMotion install does
not require Hugging Face or Civitai API tokens.

When a gated/private asset needs authentication, users may save optional Hugging
Face or Civitai credentials in WebbDuck Settings. DuckMotion CLI setup reads the
shared local `~/.webbduck/provider_credentials.json` contract directly, so
WebbDuck does not need to be running and no shell export is required. The token
contents are never part of model metadata, provenance cache records, recipes, or
job payloads.

Explicit environment variables remain advanced overrides and take precedence:

```text
HF_TOKEN
HUGGING_FACE_HUB_TOKEN
CIVITAI_TOKEN
CIVITAI_API_TOKEN
CIVITAI_API_KEY
```

`WEBBDUCK_CREDENTIALS_FILE` may override the shared credentials-file path for
advanced deployments.

## Recipe resolution

Recipe resolution may use:

1. an explicit `duckmotion_recipe.profile` declaration; or
2. a trusted adapter that maps declarative structural evidence to an installed
   profile.

Adapters must never execute arbitrary companion code/graphs. If zero profiles
match, the model is blocked with a diagnostic. If multiple profiles/recipe
candidates match, DuckMotion refuses to guess.

For LTX-2.5 ConvRot, graph topology identifies the installed two-stage AV
profile. Checkpoint-author sampling details such as Euler vs Euler ancestral and
custom sigma schedules are normalized separately. This prevents a new tuning
schedule from being mistaken for a new architecture while still preserving the
checkpoint author's intended generation behavior.

The allow-listed normalized execution settings currently include:

```text
sampler: euler | euler_ancestral
stage1_sigmas
stage2_sigmas
cfg
image_to_video stage1/stage2 guide strengths
stage2 noise policy: same_seed | increment
```

Anything missing, malformed, unsupported, or ambiguous falls back to the audited
profile default instead of being executed dynamically.

This makes the trust chain intentionally one-way:

```text
content hash identifies provenance
  -> provenance supplies declarative evidence
  -> recipe adapter proves a compatible profile
  -> companion tuning is normalized into safe fields
  -> profile selects worker/runtime topology
```

Never reverse that chain by letting a provider/model/display name choose a
profile.

## Asset providers and profile defaults

Recipes declare asset roles rather than requiring a global folder layout.
`model_asset_providers.py` normalizes architecture/format-specific discovery into
an asset manifest. Generic `tools/prepare_model_assets.py` then:

- keeps already-resolved local assets;
- searches provider/model cache roots;
- may fetch a missing asset only from a supported trusted source URL;
- may use an execution profile's trusted standard source when the recipe names
  that exact standard asset but omits its URL;
- never substitutes a profile default for a differently named custom asset;
- respects normal authentication/gating;
- never downloads or substitutes the selected checkpoint itself.

Profile-owned standard asset sources are recipe/runtime semantics, not community
checkpoint metadata. The LTX-2.5 ConvRot profile now uses current LTX-2.5 quality
assets as its standard defaults, including the full video VAE and LTX-2.5 latent
spatial upscaler.

A quality policy may upgrade only explicitly known legacy standard assets (for
example the lightweight LTX-2.5 Conv VAE or older standard spatial upscaler) to
the current profile defaults. A differently named custom companion asset is
never silently replaced. Advanced deployments that need exact legacy asset
behavior may set `DUCKMOTION_LTX_CONVROT_ASSET_POLICY=recipe`.

Memory policy stays independent of recipe fidelity. DynamicVRAM, CPU/offload
behavior, split attention, and tiled VAE decoding may trade speed for memory, but
they do not rewrite sampler/sigma/conditioning semantics.

## Adding support

Prefer the smallest extension:

- same format and same recipe topology: no runtime change; normalize companion
  tuning if present;
- same format, genuinely new topology: add an `ExecutionProfile` + worker;
- new provenance catalog: register a `CheckpointProvenanceProvider`;
- same recipe shape, new asset source/layout: extend/register an asset provider;
- genuinely new runtime API family: add a backend/runtime.

Do not add a brand-name branch to provenance, setup, discovery, the browser, or
job routing.
