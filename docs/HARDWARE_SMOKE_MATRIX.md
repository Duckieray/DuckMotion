# DuckMotion Hardware Smoke Matrix

Status: **preparation / no model-load claim yet**

Target host: NVIDIA RTX 5070 Ti 16 GB. Runtime and browser architecture are
model-driven; this document defines the first real Wan/LTX hardware validation.

## 1. Prepare DuckMotion

Normal users prepare runtimes, persist their shared model root, repair runtime
adjuncts, and prepare recipe-declared support assets with one command:

```bash
python tools/setup.py --models /path/to/models
```

DuckMotion uses deterministic runtime locations under
`~/.local/share/duckmotion/runtimes/`; `DUCKMOTION_*_PYTHON` variables are
advanced overrides only.

For targeted runtime maintenance:

```bash
python tools/prepare_model_runtimes.py all
```

Wan is pinned to Diffusers 0.39.0 and includes `gguf==0.19.0`. Standard LTX-2.5
uses the selected Diffusers-main commit. The ConvRot runtime owns a pinned Comfy
core checkout and imports it as a Python library only; no ComfyUI server/UI/API
is started.

`prepare_model_runtimes.py` never downloads model weights. The higher-level
`setup.py` may download **recipe-declared support assets** through registered
asset providers when a trusted Hugging Face source URL is present; it never
silently downloads the user's selected checkpoint.

Before model loading, run:

```bash
python tools/doctor.py
```

Doctor scans the configured model root plus the normal Hugging Face cache and
performs the same non-loading runtime/recipe/asset gates used by DuckMotion. After
starting WebbDuck, the equivalent live API surface is:

```text
GET /runtime-readiness
```

For each discovered target, `runtime.ready=true` means the worker interpreter can
import the required runtime APIs and sees CUDA. GGUF models additionally probe
the GGUF-specific Wan surface. ConvRot readiness additionally requires a
compatible execution profile and all assets declared by that recipe.

## 2. Weight preparation

### Wan2.2 TI2V-5B Diffusers — baseline

```bash
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

### Wan2.2 I2V A14B Diffusers — feasibility only

```bash
hf download Wan-AI/Wan2.2-I2V-A14B-Diffusers
```

This remains an explicit 16 GB feasibility test rather than a recommended
production checkpoint.

### Wan2.2 GGUF — regression target

Keep a compatible high/low denoiser pair together anywhere in a scanned local
model root, for example:

```text
checkpoint/wan/Wan2.2-Community/
├── Wan2.2_Community_Q8_H.gguf
└── Wan2.2_Community_Q8_L.gguf
```

DuckMotion discovers the pair as **one model**. Selecting it persists the H-file
source path; the Wan worker finds the L mate automatically.

### LTX-2.5 distilled/two-stage

```bash
hf download Lightricks/LTX-2.5-Diffusers \
  --exclude "transformer_full/*"
```

The latent upsampler must remain present.

### LTX-2.5 INT8 ConvRot — recipe/profile target

A ConvRot checkpoint is only the checkpoint format. It additionally needs a
companion recipe that resolves to an installed execution profile.

DuckMotion currently ships:

```text
ltx25_convrot_two_stage_av
```

The profile was audited from a REDGraft workflow, so the current REDGraft
checkpoint remains our first real hardware fixture. It is **not** the routing
key: any unrelated LTX-2.5 ConvRot checkpoint with a compatible recipe may be
used for the same rows.

A companion recipe may be a native `duckmotion_recipe` manifest or a supported
exported workflow that DuckMotion can map structurally to the profile. Required
support assets may live anywhere under the configured model root or DuckMotion
asset cache.

## 3. Smoke order

| Order | Target | Workflow | Canary | Reference/default gate |
|---|---|---|---|---|
| 1 | Wan2.2 TI2V-5B | T2V | 640x384, 33 frames, seed 0 | 1280x704, 121 frames |
| 2 | Wan2.2 paired GGUF | advertised T2V/I2V | 512-640px short clip, seed 0 | prove H+L load without stock transformer download |
| 3 | Wan2.2 I2V A14B BF16 | I2V | 512x320, 17 frames | only if 16 GB path is viable |
| 4 | LTX-2.5 Diffusers | T2V + audio | 768x512, 33 frames | 1536x1024, 121 frames |
| 5 | LTX-2.5 Diffusers | I2V + audio | same canary with source image | reference-size only after T2V passes |
| 6 | LTX-2.5 INT8 ConvRot | T2V + audio | 768x512, 33 frames | resolved profile defaults |
| 7 | LTX-2.5 INT8 ConvRot | I2V + audio | same canary with source image | resolved profile defaults after T2V passes |

Canary dimensions still obey model constraints: Wan uses dimensions divisible by
16 and `4k+1` frames; LTX uses dimensions divisible by 64 and `8k+1` frames.

For `ltx25_convrot_two_stage_av`, the current profile defaults are 1152x768,
241 frames, 24 fps. Those are profile defaults rather than ConvRot-format
defaults. The smoke runner obtains them from runtime readiness metadata.

## 4. API-driven runner

Safe preflight:

```bash
python tools/run_hardware_smoke.py
```

The runner identifies ConvRot using readiness `source_format=int8_convrot`, not
checkpoint/model names.

Run the practical ConvRot canaries:

```bash
python tools/run_hardware_smoke.py \
  --execute \
  --only ltx25-convrot-t2v-canary \
  --only ltx25-convrot-i2v-canary
```

If multiple ConvRot checkpoints are discovered, an explicit public model identity
can still be supplied as a test override:

```bash
python tools/run_hardware_smoke.py \
  --execute \
  --ltx-convrot-model "/path/to/community-checkpoint.safetensors" \
  --only ltx25-convrot-t2v-canary
```

After the canaries pass, opt into the heavy/default-profile row:

```bash
python tools/run_hardware_smoke.py \
  --execute --include-heavy \
  --only ltx25-convrot-default
```

The payload for that row is built from `runtime.profile_defaults` rather than a
model-name check.

The Wan row names predate restored GGUF discovery; the selected public
model/source determines execution. GGUF execution itself must not depend on UI or
test naming.

## 5. Pass conditions

### Wan

A row passes only when:

- the isolated worker sees the RTX 5070 Ti;
- the correct T2V or I2V pipeline loads;
- GGUF rows load the local H/L pair with `from_single_file()`;
- GGUF rows do **not** fetch the stock second transformer when the L mate exists;
- requested conditioning is honored;
- MP4/poster/metadata are written;
- the worker exits and GPU lease/memory are released.

### Standard LTX-2.5

A row additionally requires stage-1 latent generation, 2x latent upsample,
stage-2 distilled refinement, synchronized audio, and normalized final output.

### LTX-2.5 ConvRot

A row additionally requires:

- checkpoint format recognized independently of recipe selection;
- a single compatible execution profile resolved without ambiguity;
- all profile-declared support assets resolved before model loading;
- pinned Comfy core imports without starting a Comfy server;
- all nodes required by the **selected profile** available;
- the selected profile worker receives profile-owned defaults;
- synchronized decoded audio/video in the output artifact;
- successful worker teardown and GPU lease release.

For the current `ltx25_convrot_two_stage_av` profile, validation additionally
checks its fixed two-stage AV sigma/guide/upscale/decode contract documented in
`docs/LTX25_CONVROT.md`.

## 6. Failure classification

Classify failures before changing generic architecture:

1. **runtime** — missing import/version/CUDA/GGUF/Comfy package or runtime-owned checkout;
2. **format** — checkpoint architecture/format cannot be identified safely;
3. **recipe** — no compatible profile, ambiguous profile evidence, or unsupported manifest;
4. **assets** — missing/gated/incomplete snapshot, missing GGUF mate, or recipe-declared support asset unavailable;
5. **memory** — VRAM/host-RAM/offload failure;
6. **pipeline API** — upstream Diffusers or pinned-Comfy behavior/signature mismatch;
7. **adapter** — DuckMotion request/result/pair/recipe translation bug;
8. **model behavior** — generation completes but output/conditioning/audio is invalid.

Do not expose Wan/LTX/runtime/format/profile selectors to solve a failure. Fix the
responsible descriptor/provider/profile/backend/runtime instead.
