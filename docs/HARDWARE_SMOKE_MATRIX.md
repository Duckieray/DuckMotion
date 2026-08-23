# DuckMotion Hardware Smoke Matrix

Status: **preparation / no model-load claim yet**

Target host: NVIDIA RTX 5070 Ti 16 GB. Runtime and browser architecture are
model-driven; this document defines the first real Wan/LTX hardware validation.

## 1. Prepare DuckMotion

Normal users should prepare runtimes and persist their shared model root with one
command:

```bash
python tools/setup.py --models /path/to/models
```

The setup command creates the isolated Wan, standard LTX, and ConvRot runtimes at
the deterministic `~/.local/share/duckmotion/runtimes/` location. DuckMotion
finds those interpreters automatically; `DUCKMOTION_*_PYTHON` variables are
advanced overrides only.

For targeted runtime maintenance the lower-level command remains available:

```bash
python tools/prepare_model_runtimes.py all
```

Wan is pinned to Diffusers 0.39.0 and includes `gguf==0.19.0` so both normal
Diffusers checkpoints and the restored hybrid Diffusers + GGUF path execute in
the same isolated Wan runtime. Standard LTX-2.5 is pinned to the selected
Diffusers-main commit because its APIs have not yet landed in a stable release.
The ConvRot runtime owns a separate pinned Comfy core checkout and imports it as
a Python library only; no ComfyUI server/UI/API is started.

Runtime setup never downloads model weights.

Before model loading, run:

```bash
python tools/doctor.py
```

Doctor scans the configured model root plus the normal Hugging Face cache and
performs the same non-loading runtime/asset gates used by DuckMotion. After
starting WebbDuck, the equivalent live API surface is:

```text
GET /runtime-readiness
```

For each discovered target, `runtime.ready=true` means the worker interpreter
can import the required runtime APIs and sees CUDA. GGUF models additionally
probe `WanTransformer3DModel`, `GGUFQuantizationConfig`, and the `gguf` package.
ConvRot readiness additionally requires the REDGraft companion JSON plus its
text encoder, latent upscaler, video VAE, and audio VAE. This is an environment
and asset gate, not a generation success claim.

## 2. Weight preparation

### Wan2.2 TI2V-5B Diffusers — baseline

```bash
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

This is the first plain-Diffusers Wan target. DuckMotion currently exposes its
text-to-video path.

### Wan2.2 I2V A14B Diffusers — feasibility only

```bash
hf download Wan-AI/Wan2.2-I2V-A14B-Diffusers
```

This is very large and remains an explicit 16 GB feasibility test rather than a
recommended production checkpoint.

### Wan2.2 GGUF — regression target

Keep the compatible high/low denoiser pair together in any local model root
scanned by DuckMotion, for example:

```text
checkpoint/wan/Wan2.2-Enhanced-NSFW-I2V-T2V/
├── Wan2.2_Enhanced_NSFW_I2V_T2V_Q8_H.gguf
└── Wan2.2_Enhanced_NSFW_I2V_T2V_Q8_L.gguf
```

DuckMotion should discover that pair as **one model**. Selecting it persists the
H-file source path; the Wan worker finds the L mate automatically and injects
both transformers into the normal Diffusers Wan pipeline. Do not add a UI
backend selector or separate GGUF configuration field.

For T2V, the default component source is
`Wan-AI/Wan2.2-T2V-A14B-Diffusers`; for I2V it is
`Wan-AI/Wan2.2-I2V-A14B-Diffusers`. Those repos supply scheduler/VAE/text
encoder/config components, while the local GGUF files replace the denoisers.

### LTX-2.5 distilled/two-stage

```bash
hf download Lightricks/LTX-2.5-Diffusers \
  --exclude "transformer_full/*"
```

The latent upsampler must remain present.

### REDGraft LTX-2.5 INT8 ConvRot

Keep the checkpoint companion JSON and its declared support files within the
selected checkpoint directory or configured model roots:

```text
REDGraft-ltx25-sulphur2-int8-convrot-ComfyMCP.safetensors
redgraftLTX25Fast2K_ltx25RedgraftNSFW.json
gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
ltx-2.3-spatial-upscaler-x2-1.1.safetensors
ltx-2.5-video-vae-conv-bf16.safetensors
ltx-2.5-audio-vae-bf16.safetensors
```

DuckMotion should discover the REDGraft checkpoint as one normal public model.
The checkpoint does not need to be renamed merely to contain the word `convrot`;
REDGraft identity and safetensors metadata are part of detection. The
companion/support files are runtime dependencies, not separate user model
selections.

## 3. Smoke order

| Order | Target | Workflow | Canary | Reference/default gate |
|---|---|---|---|---|
| 1 | Wan2.2 TI2V-5B | T2V | 640x384, 33 frames, seed 0 | 1280x704, 121 frames |
| 2 | Wan2.2 paired GGUF | advertised T2V/I2V | 512-640px short clip, seed 0 | prove H+L load without stock transformer download |
| 3 | Wan2.2 I2V A14B BF16 | I2V | 512x320, 17 frames | only if 16 GB path is viable |
| 4 | LTX-2.5 Diffusers | T2V + audio | 768x512, 33 frames | 1536x1024, 121 frames |
| 5 | LTX-2.5 Diffusers | I2V + audio | same canary with source image | reference-size only after T2V passes |
| 6 | REDGraft LTX-2.5 ConvRot | T2V + audio | 768x512, 33 frames | saved recipe 1152x768, 241 frames |
| 7 | REDGraft LTX-2.5 ConvRot | I2V + audio | same canary with source image | saved-recipe size only after T2V passes |

Canary dimensions still obey model constraints: Wan uses dimensions divisible by
16 and `4k+1` frames; both LTX runtimes use final dimensions divisible by 64 and
`8k+1` frames. Standard LTX keeps its explicit distilled sigma schedule even for
canaries. ConvRot keeps its fixed REDGraft high/low sigma schedules, Euler, CFG
1, AV latent path, guide cropping, and latent-upscale chain; canaries only reduce
resolution/frame count.

The REDGraft top-level workflow controls store duration `10`, width `1152`,
height `768`, and frame rate `24`. Its linked length expression is
`duration * frame_rate + 1`, so the reference gate is `10 * 24 + 1 = 241`
frames. Group-local 97-frame / 25-fps widget values are overridden by these
linked top-level controls and are not the effective saved recipe.

## 4. API-driven runner

Safe preflight:

```bash
python tools/run_hardware_smoke.py
```

The runner distinguishes standard LTX from ConvRot so a REDGraft checkpoint
cannot accidentally satisfy the Diffusers LTX smoke target.

Run the two practical REDGraft canaries explicitly:

```bash
python tools/run_hardware_smoke.py \
  --execute \
  --ltx-convrot-model "/path/to/REDGraft.safetensors" \
  --only ltx25-convrot-t2v-canary \
  --only ltx25-convrot-i2v-canary
```

After both canaries pass, opt into the heavier saved-recipe row:

```bash
python tools/run_hardware_smoke.py \
  --execute --include-heavy \
  --ltx-convrot-model "/path/to/REDGraft.safetensors" \
  --only ltx25-convrot-default
```

The runner can also target a discovered GGUF source explicitly through the
existing Wan override arguments. For a T2V-capable GGUF checkpoint:

```bash
python tools/run_hardware_smoke.py \
  --execute \
  --wan-5b-model "/path/to/...H.gguf" \
  --only wan-ti2v-5b-canary
```

For an I2V-capable GGUF checkpoint:

```bash
python tools/run_hardware_smoke.py \
  --execute --include-heavy \
  --wan-i2v-model "/path/to/...H.gguf" \
  --only wan-i2v-a14b-canary
```

The Wan row names predate restored GGUF discovery; the selected public
model/source is what determines execution. GGUF execution itself must not depend
on UI/test naming.

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

### REDGraft LTX-2.5 ConvRot

A row additionally requires:

- strict companion/support-asset readiness before model loading;
- pinned Comfy core imports without starting a Comfy server;
- video/audio latent concatenation for both sampling stages;
- exact fixed high/low REDGraft sigma schedules, Euler, CFG 1;
- `LTXVCropGuides` feeding stage-two conditioning;
- learned latent upsampler followed by bicubic 0.5 resize;
- I2V guide strength 0.7 in stage one and 1.0 after upscale;
- synchronized decoded audio/video in the output artifact;
- successful worker teardown and GPU lease release.

## 6. Failure classification

Classify failures before changing generic architecture:

1. **runtime** — missing import/version/CUDA/GGUF/Comfy package;
2. **weights** — missing/gated/incomplete snapshot, missing GGUF mate, or missing ConvRot companion/support asset;
3. **memory** — VRAM/host-RAM/offload failure;
4. **pipeline API** — upstream Diffusers or pinned-Comfy behavior/signature mismatch;
5. **adapter** — DuckMotion request/result/pair/recipe translation bug;
6. **model behavior** — generation completes but output/conditioning/audio is invalid.

Do not expose Wan/LTX/runtime/format selectors to solve a failure. Fix the
responsible descriptor/backend/runtime instead.
