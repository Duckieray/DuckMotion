# LTX-2.5 INT8 ConvRot

Status: **format/runtime/profile/provenance contracts available; real-model GPU smoke validation pending**

DuckMotion treats LTX-2.5 INT8 ConvRot as a checkpoint **format**, not as a model
brand or a sampling recipe. Users select the checkpoint normally; there is no
ConvRot, Comfy, provenance, backend, or recipe selector in the product UI.

## Separation of concerns

DuckMotion keeps these contracts separate:

```text
Checkpoint descriptor
  architecture: ltx25
  source_format: int8_convrot

Checkpoint provenance
  strong content fingerprint
  canonical source/version identity
  cached declarative recipe candidates

Execution profile
  profile id
  worker entrypoint
  required runtime nodes
  required asset roles + trusted standard sources
  defaults / constraints

Display identity
  arbitrary model/community name
```

A checkpoint being ConvRot does not automatically mean it should use one
particular resolution, sigma schedule, asset set, or workflow. Those belong to
the resolved execution profile. Provenance can identify the exact file and
supply declarative evidence, but provenance names/IDs never select that profile.

The current installed profile is:

```text
ltx25_convrot_two_stage_av
```

Its implementation was audited from a known REDGraft workflow. REDGraft is an
important compatibility and hardware-test fixture for that profile, **not** a
routing key. An unrelated ConvRot checkpoint may use the same profile if its
companion recipe declares or structurally matches it.

## Runtime model

The ConvRot runtime uses its dedicated interpreter under the normal deterministic
runtime root:

```text
~/.local/share/duckmotion/runtimes/ltx25_convrot/bin/python
```

Normal users do not need to export an interpreter path. The advanced override is:

```bash
DUCKMOTION_LTX_CONVROT_PYTHON
```

The runtime owns a detached Comfy core checkout pinned to:

```text
9db05e0e1f035d1902ffc256fe7a336e549ced34
```

The checkout lives under `ltx25_convrot/comfyui` unless the advanced
`DUCKMOTION_LTX_CONVROT_COMFY_ROOT` override is used.

DuckMotion imports that checkout directly as a Python library. It does **not**:

- start a ComfyUI server;
- expose the Comfy web UI;
- use the Comfy HTTP API;
- execute a companion JSON as arbitrary code.

`tools/setup.py` repairs this runtime-owned checkout even when package
reinstallation is skipped.

## Provenance resolution

A recognized ConvRot checkpoint with no usable local companion recipe may be
identified during setup by a registered `CheckpointProvenanceProvider`.

The first built-in provider uses Civitai's public model-version-by-hash API:

```text
local checkpoint
  -> full SHA256
  -> Civitai model version by hash
  -> verify returned files include that exact SHA256
  -> cache model-version metadata
  -> cache small published JSON sidecar candidates
```

This is setup-time trust work only. DuckMotion never re-downloads or replaces the
selected checkpoint. The fingerprint and provenance record are stored under
`~/.cache/duckmotion/provenance/` by default and tied to the local file's change
state. A changed checkpoint invalidates the old record.

Once recipe JSON has been cached, doctor/readiness do not contact Civitai or
re-hash the unchanged checkpoint. `CIVITAI_API_TOKEN` is only an optional
advanced credential if access to a published sidecar requires it.

Exactly one provenance provider must match. Zero matches leave the checkpoint
unresolved; multiple matches are treated as ambiguous. Even one valid provenance
match does **not** select an execution profile. Its cached JSON still has to pass
the same recipe adapters as a local companion.

## Recipe resolution

A recognized ConvRot checkpoint is runnable only after a local or
provenance-cached companion recipe maps to an installed execution profile.

DuckMotion accepts two recipe inputs today.

### Native DuckMotion manifest

A companion may explicitly name a profile and its asset roles:

```json
{
  "checkpoint": "SomeCommunityLTX25ConvRot.safetensors",
  "duckmotion_recipe": {
    "profile": "ltx25_convrot_two_stage_av",
    "assets": {
      "text_encoder": {
        "name": "encoder.safetensors",
        "url": "https://huggingface.co/org/repo/resolve/main/encoder.safetensors"
      },
      "latent_upscaler": {
        "name": "upscaler.safetensors"
      },
      "video_vae": {
        "name": "video-vae.safetensors"
      },
      "audio_vae": {
        "name": "audio-vae.safetensors"
      }
    }
  }
}
```

The manifest does not contain executable code. Profile IDs must already be
registered by DuckMotion.

### Exported-workflow adapter

For existing community bundles, DuckMotion can inspect an exported workflow as
declarative evidence. It extracts node types and declared model records, then
maps that evidence to a registered execution profile.

The workflow itself is never executed. If zero profiles or multiple profiles
match the evidence, DuckMotion refuses to guess. If multiple companion candidates
independently map to supported profiles, DuckMotion also refuses to pick one
arbitrarily.

## Asset preparation

`tools/prepare_model_assets.py` is provider-driven. Setup does not branch on
REDGraft, ConvRot filenames, or other model brands.

A provider normalizes a recipe into asset roles such as:

```text
text_encoder
latent_upscaler
video_vae
audio_vae
```

DuckMotion first searches the checkpoint directory, inferred/configured shared
model roots, and its own asset cache recursively. Assets do not need a Comfy-style
folder layout.

The current two-stage AV execution profile owns trusted standard sources for:

- `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` from
  `Lightricks/LTX-2.5`;
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` from `Lightricks/LTX-2.3`;
- `ltx-2.5-video-vae-conv-bf16.safetensors` from `Lightricks/LTX-2.5`;
- `ltx-2.5-audio-vae-bf16.safetensors` from `Lightricks/LTX-2.5`.

A profile default can fill a missing asset role, or a missing source URL for the
same standard filename. It **cannot** replace a differently named custom asset
from a recipe.

Missing supported assets are fetched through the owning runtime's normal
`huggingface_hub` client. Gated repositories remain gated: DuckMotion does not
accept terms or bypass authentication on the user's behalf.

A format can therefore be supported while an individual checkpoint remains
blocked because provenance is missing/ambiguous, its recipe is missing,
ambiguous, unsupported, or its assets are unavailable. `doctor.py` reports those
distinctions before model loading.

## Current two-stage AV profile

`ltx25_convrot_two_stage_av` currently owns these defaults:

```text
final resolution: 1152x768
fps:              24
frames:           241
CFG:              1.0
sampler:          Euler
```

These are **profile defaults**, not ConvRot-format defaults. Checkpoint discovery
therefore does not inject them merely because a file is INT8 ConvRot.

Final dimensions are snapped to multiples of 64 and frame counts to `8k+1`.
Stage one runs at half final width/height.

### Conditioning

The current profile uses positive LTX/Gemma conditioning and
`ConditioningZeroOut(positive)` for the negative path, followed by
`LTXVConditioning` at the requested frame rate.

For I2V, the source image is resized with Lanczos to a 1536-pixel longer edge and
processed with:

```text
LTXVPreprocess(img_compression=18)
```

The stage-one image guide strength is `0.7`.

### Stage one

Video and audio latents are created separately and concatenated with
`LTXVConcatAVLatent` before sampling.

High-noise sigma schedule:

```text
1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0
```

After sampling, `LTXVSeparateAVLatent` separates video and audio again.

### Upscale and stage two

The current profile applies:

```text
LTXVCropGuides
  -> LTXVLatentUpsampler
  -> LatentUpscaleBy(upscale_method="bicubic", scale_by=0.5)
  -> LTXVImgToVideoInplace(strength=1.0)  # I2V only
```

The upscaled video latent is recombined with audio for the second pass.

Low-noise sigma schedule:

```text
0.85, 0.7250, 0.4219, 0.0
```

DuckMotion's single public seed deterministically derives the second noise stream
as `seed + 1` modulo uint64.

### Decode and output

The final video latent is tiled-decoded with:

```text
VAEDecodeTiled(
  tile_size=480,
  overlap=96,
  temporal_size=96,
  temporal_overlap=24
)
```

Audio is decoded through `LTXVAudioVAEDecode`. The worker writes H.264 MP4 with
CRF 16 through pinned Comfy core `CreateVideo` + `SaveVideo`.

## Extending ConvRot

A future ConvRot recipe should normally require **no new backend and no new UI
mode**.

Add an `ExecutionProfile` that declares:

- compatible architecture + source format;
- worker entrypoint;
- structural evidence used by workflow adapters;
- required runtime nodes;
- required asset roles and any trusted standard sources;
- profile defaults/constraints.

If a new public catalog can identify exact checkpoints, register a generic
`CheckpointProvenanceProvider`. If its asset shape needs different
interpretation, extend/register an asset provider. Generic setup continues to
call only `prepare_model_assets.py`.

Never let provenance provider model/version names select a profile. If two
providers, profiles, or supported recipe candidates are ambiguous, refuse to
guess.

## Validation

Unit/contract coverage protects:

- profile-owned defaults, standard asset sources, and runtime-node requirements;
- checkpoint format detection independent of recipe selection;
- strong checkpoint provenance caching/invalidation;
- exact-SHA provenance validation and ambiguity refusal;
- explicit DuckMotion manifests;
- exported-workflow adaptation;
- generic asset-provider iteration;
- arbitrary shared-model layouts;
- no brand-specific recipe table;
- ambiguous profile/recipe evidence refusing to guess;
- the current two-stage AV operation ordering and constants.

`tools/run_hardware_smoke.py` identifies ConvRot models using readiness
`source_format` metadata, not model names, and obtains heavy/default dimensions
from the resolved profile.

A passing test suite is **not** hardware validation. Real T2V/I2V generation still
has to pass on target hardware with an actual checkpoint, resolved recipe/assets,
synchronized audio/video, successful teardown, and acceptable memory behavior.
