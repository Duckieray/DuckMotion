# DuckMotion Hardware Smoke Matrix

Status: **preparation / no model-load claim yet**

Target host: NVIDIA RTX 5070 Ti 16 GB. Runtime and browser architecture are
model-driven; this document defines the first real Wan/LTX hardware validation.

## 1. Prepare isolated runtimes

From a shell with Python/venv support (on NixOS, `nix develop` is the intended
entry point):

```bash
python tools/prepare_model_runtimes.py all
```

The tool defaults to PyTorch 2.12.1 from the CUDA 13.0 wheel channel and prints:

```bash
export DUCKMOTION_WAN_PYTHON=...
export DUCKMOTION_LTX_PYTHON=...
```

Wan is pinned to Diffusers 0.39.0. LTX-2.5 is not in a stable Diffusers release
yet, so its environment is pinned to the exact Diffusers-main commit selected
for this smoke baseline rather than floating with `main`.

Runtime setup never downloads model weights.

Before model loading, start WebbDuck/DuckMotion and check the plugin's:

```text
GET /runtime-readiness
```

For each discovered target, `runtime.ready=true` means the worker interpreter
can import the exact required pipeline APIs and that interpreter sees CUDA. The
payload also reports its PyTorch/Diffusers versions, GPU name, compute capability
and VRAM. It is an environment gate, not a generation success claim.

## 2. Weight preparation

Use the normal Hugging Face cache.

### Wan2.2 TI2V-5B Diffusers — first Wan target

```bash
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

This package is roughly 34 GB, dramatically smaller than the A14B I2V package.
DuckMotion's current Diffusers backend can run its text-to-video path. Upstream
TI2V image conditioning remains known-but-not-runnable until Diffusers/native
runtime support catches up.

### Wan2.2 I2V A14B Diffusers — feasibility target

```bash
hf download Wan-AI/Wan2.2-I2V-A14B-Diffusers
```

This package is roughly 126 GB. Do not interpret discovery/readiness as a claim
that raw BF16 A14B will be practical on a 16 GB GPU. The first load is explicitly
a feasibility test of group/sequential/disk offload; a later quantized runtime
may be the correct production path.

### LTX-2.5 distilled/two-stage

Accept the gated model terms first, then:

```bash
hf download Lightricks/LTX-2.5-Diffusers \
  --exclude "transformer_full/*"
```

The distilled path is the default `transformer/` in `model_index.json`; the full
SFT transformer is separate and not used by DuckMotion's two-stage distilled
backend. Excluding it reduces a snapshot from roughly 110 GB to roughly 72 GB.
The latent upsampler must remain present.

## 3. Smoke order

Use a cheap canary first, then the checkpoint's normal/reference settings. The
purpose of a canary is to validate loading, conditioning, artifact writing and
cleanup without spending the full reference runtime on a broken setup.

| Order | Target | Workflow | Canary | Reference gate |
|---|---|---|---|---|
| 1 | Wan2.2 TI2V-5B | T2V | 640x384, 17 frames, seed 0 | 1280x704, 121 frames, 24 fps, published/model defaults |
| 2 | Wan2.2 I2V A14B | I2V | small source, 512-ish resolution, 17 frames, seed 0 | only attempt normal settings if the 16 GB load/offload path is viable |
| 3 | LTX-2.5 | T2V + audio | final 768x512, 33 frames, 24 fps, seed 0 | final 1536x1024, 121 frames, two-stage distilled schedule |
| 4 | LTX-2.5 | I2V + audio | same canary dimensions with a source image | reference-size I2V only after T2V passes |

Canary dimensions still obey each descriptor's constraints:

- Wan dimensions divisible by 16; frame count `4k+1`;
- LTX final dimensions divisible by 64; frame count `8k+1`.

For LTX, do **not** replace the explicit distilled sigma schedules with an
arbitrary low step count for the canary. Reduce dimensions/frames instead; the
sampling schedule is model semantics.

## 4. Pass conditions

### Wan

A row passes only when:

- the isolated worker actually sees the RTX 5070 Ti;
- the correct T2V or I2V pipeline loads;
- requested conditioning is honored;
- MP4 and poster are written;
- job/gallery metadata records the selected public model and normalized params;
- cancellation/job state remains functional;
- the worker exits and GPU memory/lease are released.

### LTX-2.5

A row additionally requires:

- stage-1 latent generation;
- 2x latent spatial upsample;
- stage-2 distilled-sigma refinement;
- synchronized audio produced and muxed into the output;
- final dimensions/frame count match normalized model constraints.

## 5. What to record

For every run record:

- public selected-model name;
- worker Python, PyTorch and Diffusers version;
- GPU name/compute capability/VRAM;
- checkpoint source/cache location;
- load time and generation time;
- peak host RAM if available;
- offload mode/fallbacks;
- requested vs normalized dimensions/frames/FPS/seed;
- output video/poster/metadata paths;
- audio presence/sample rate for LTX;
- CUDA memory state after worker exit/model switch.

## 6. Failure classification

Classify failures before changing generic architecture:

1. **runtime** — missing import/version/CUDA;
2. **weights** — gated/missing/incomplete snapshot;
3. **memory** — VRAM/host-RAM/offload failure;
4. **pipeline API** — upstream Diffusers behavior/signature drift;
5. **adapter** — DuckMotion request/result translation bug;
6. **model behavior** — generation completes but output/conditioning is invalid.

Do not expose Wan/LTX/runtime selectors to solve a failure. Fix the responsible
backend/runtime or adjust the model's public capabilities/constraints.
