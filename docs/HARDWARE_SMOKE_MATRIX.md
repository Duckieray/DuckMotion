# DuckMotion Hardware Smoke Matrix

Status: **preparation / no model-load claim yet**

Target host: NVIDIA RTX 5070 Ti 16 GB. Runtime and browser architecture are
model-driven; this document defines the first real Wan/LTX hardware validation.

## 1. Prepare isolated runtimes

From any shell with normal Python/venv support:

```bash
python tools/prepare_model_runtimes.py all
```

The tool defaults to PyTorch 2.12.1 from the CUDA 13.0 wheel channel and prints:

```bash
export DUCKMOTION_WAN_PYTHON=...
export DUCKMOTION_LTX_PYTHON=...
```

Wan is pinned to Diffusers 0.39.0 and includes `gguf==0.19.0` so both normal
Diffusers checkpoints and the restored hybrid Diffusers + GGUF path execute in
the same isolated Wan runtime. LTX-2.5 is pinned to the selected Diffusers-main
commit because its APIs have not yet landed in a stable release.

Runtime setup never downloads model weights.

Before model loading, start WebbDuck/DuckMotion and check:

```text
GET /runtime-readiness
```

For each discovered target, `runtime.ready=true` means the worker interpreter
can import the required runtime APIs and sees CUDA. GGUF models additionally
probe `WanTransformer3DModel`, `GGUFQuantizationConfig`, and the `gguf` package.
This is an environment gate, not a generation success claim.

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

## 3. Smoke order

| Order | Target | Workflow | Canary | Reference gate |
|---|---|---|---|---|
| 1 | Wan2.2 TI2V-5B | T2V | 640x384, 33 frames, seed 0 | 1280x704, 121 frames |
| 2 | Wan2.2 paired GGUF | advertised T2V/I2V | 512-640px short clip, seed 0 | prove H+L load without stock transformer download |
| 3 | Wan2.2 I2V A14B BF16 | I2V | 512x320, 17 frames | only if 16 GB path is viable |
| 4 | LTX-2.5 | T2V + audio | 768x512, 33 frames | 1536x1024, 121 frames |
| 5 | LTX-2.5 | I2V + audio | same canary with source image | reference-size only after T2V passes |

Canary dimensions still obey model constraints: Wan uses dimensions divisible by
16 and `4k+1` frames; LTX uses final dimensions divisible by 64 and `8k+1`
frames. LTX keeps its explicit distilled sigma schedule even for canaries.

## 4. API-driven runner

Safe preflight:

```bash
python tools/run_hardware_smoke.py
```

The runner can already target a discovered GGUF source explicitly through the
existing Wan override arguments while this regression is being validated. For a
T2V-capable GGUF checkpoint:

```bash
python tools/run_hardware_smoke.py \
  --execute \
  --wan-5b-model "/path/to/..._H.gguf" \
  --only wan-ti2v-5b-canary
```

For an I2V-capable GGUF checkpoint:

```bash
python tools/run_hardware_smoke.py \
  --execute --include-heavy \
  --wan-i2v-model "/path/to/..._H.gguf" \
  --only wan-i2v-a14b-canary
```

The row names predate restored GGUF discovery; the selected public model/source
is what determines execution. A follow-up runner cleanup may rename/add dedicated
GGUF rows after the first real hardware result, but GGUF execution itself must
not depend on such UI/test naming.

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

### LTX-2.5

A row additionally requires stage-1 latent generation, 2x latent upsample,
stage-2 distilled refinement, synchronized audio, and normalized final output.

## 6. Failure classification

Classify failures before changing generic architecture:

1. **runtime** — missing import/version/CUDA/GGUF package;
2. **weights** — missing/gated/incomplete snapshot or missing GGUF pair mate;
3. **memory** — VRAM/host-RAM/offload failure;
4. **pipeline API** — upstream Diffusers behavior/signature drift;
5. **adapter** — DuckMotion request/result/pair translation bug;
6. **model behavior** — generation completes but output/conditioning is invalid.

Do not expose Wan/LTX/runtime/format selectors to solve a failure. Fix the
responsible descriptor/backend/runtime instead.
