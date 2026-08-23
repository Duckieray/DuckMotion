# LTX-2.5 INT8 ConvRot / REDGraft

Status: **implementation and contract tests available; real-model GPU smoke validation pending**

DuckMotion supports REDGraft-style LTX-2.5 INT8 ConvRot checkpoints as a private
runtime variant. Users still select the checkpoint normally; there is no Comfy or
backend selector in the product UI.

## Runtime model

The ConvRot worker uses a dedicated Python environment selected by:

```bash
DUCKMOTION_LTX_CONVROT_PYTHON
```

Prepare it with:

```bash
python tools/prepare_model_runtimes.py ltx25_convrot
```

The helper installs the ConvRot requirements, PyTorch/TorchAudio, and a detached
Comfy core checkout pinned to:

```text
9db05e0e1f035d1902ffc256fe7a336e549ced34
```

The checkout lives beside the virtual environment under `ltx25_convrot/comfyui`
unless `DUCKMOTION_LTX_CONVROT_COMFY_ROOT` overrides it.

DuckMotion imports that checkout directly as a Python library. It does **not**:

- start a ComfyUI server;
- expose the Comfy web UI;
- use the Comfy HTTP API;
- execute the companion JSON as an arbitrary workflow.

Readiness initializes the same pinned builtin node registry used by the worker
and verifies the required REDGraft node surface before any model load. This is in
addition to Python/CUDA and asset checks.

## Required files

A REDGraft checkpoint needs its companion JSON and these four support weights:

```text
REDGraft-ltx25-sulphur2-int8-convrot-ComfyMCP.safetensors
redgraftLTX25Fast2K_ltx25RedgraftNSFW.json

gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
ltx-2.3-spatial-upscaler-x2-1.1.safetensors
ltx-2.5-video-vae-conv-bf16.safetensors
ltx-2.5-audio-vae-bf16.safetensors
```

They do not need to share one directory. DuckMotion searches the selected
checkpoint directory and configured model roots recursively, so a normal layout
is sufficient:

```text
models/
├── checkpoints/
│   └── ltx/
│       ├── REDGraft-ltx25-sulphur2-int8-convrot-ComfyMCP.safetensors
│       └── redgraftLTX25Fast2K_ltx25RedgraftNSFW.json
├── text_encoders/
│   └── gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
├── vae/
│   ├── ltx-2.5-video-vae-conv-bf16.safetensors
│   └── ltx-2.5-audio-vae-bf16.safetensors
└── latent_upscale_models/
    └── ltx-2.3-spatial-upscaler-x2-1.1.safetensors
```

For conventional `models/checkpoints/...` layouts, readiness infers the shared
`models/` root so sibling text-encoder/VAE/upscaler directories resolve even
though backend readiness does not receive persisted config.

Readiness fails before model loading if the companion JSON is absent, does not
declare one of the support weights, or a declared weight cannot be resolved.

## Fixed REDGraft recipe

The companion JSON is evidence for the supported recipe, but the worker executes
a fixed DuckMotion implementation.

The saved top-level REDGraft controls are:

```text
duration:         10 seconds
final resolution: 1152x768
fps:              24
frames:           241
CFG:              1.0
sampler:          Euler
```

The workflow computes latent length as `duration * fps + 1`, so the saved
10-second / 24-fps setup produces `10 * 24 + 1 = 241` frames. Some nodes inside
the grouped workflow retain stale local widget values such as 97 frames / 25 fps
or 768x512 dimensions, but their inputs are linked to the top-level controls and
those widget values are not the effective saved recipe.

Final dimensions are snapped to multiples of 64 and frame counts to `8k+1`.
Stage one receives the linked width/height divided by two, so the saved 1152x768
recipe starts at 576x384 before the learned spatial refinement path.

### Conditioning

The Gemma 4 12B ConvRot encoder produces positive conditioning. Negative
conditioning is `ConditioningZeroOut(positive)`, followed by `LTXVConditioning`
with the requested frame rate.

For I2V, the source image is first resized with Lanczos so its longer edge is
1536 pixels, then processed with:

```text
LTXVPreprocess(img_compression=18)
```

The stage-one image guide strength is `0.7`.

### Stage one

Video and audio latents are created separately. `LTXVEmptyLatentAudio` receives
the loaded LTX-2.5 audio VAE. The two latents are concatenated with
`LTXVConcatAVLatent` and sampled together.

High-noise sigma schedule:

```text
1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0
```

After sampling, `LTXVSeparateAVLatent` splits video and audio again.

### Upscale and stage two

Before upscaling, `LTXVCropGuides` returns the conditioning and video latent that
must feed stage two. The video latent then follows this exact chain:

```text
LTXVLatentUpsampler
  -> LatentUpscaleBy(upscale_method="bicubic", scale_by=0.5)
  -> LTXVImgToVideoInplace(strength=1.0)  # I2V only
```

The learned LTX 2.3 upsampler plus the 0.5 bicubic resize reproduces the net 2x
spatial refinement path in the saved REDGraft graph.

The upscaled video latent is recombined with the audio latent and sampled with
the cropped guide conditioning.

Low-noise sigma schedule:

```text
0.85, 0.7250, 0.4219, 0.0
```

The saved workflow has separate randomized `RandomNoise` nodes for stages one and
two. DuckMotion keeps its normal single-seed public API and deterministically
uses `seed + 1` modulo uint64 for the second stage. This preserves reproducibility
while keeping the two REDGraft noise streams distinct.

### Decode and output

The final AV latent is separated. Video is decoded with the saved tiled VAE
settings:

```text
VAEDecodeTiled(
  tile_size=480,
  overlap=96,
  temporal_size=96,
  temporal_overlap=24
)
```

Audio is decoded through `LTXVAudioVAEDecode` with the LTX-2.5 audio VAE.

The reference graph uses VideoHelperSuite for H.264 MP4, yuv420p, CRF 16. The
DuckMotion worker uses pinned Comfy core `CreateVideo` + `SaveVideo` instead so
no custom output-node package is required. Core Comfy's 8-bit H.264 encoder uses
yuv420p, and DuckMotion explicitly sets CRF 16.

## Why both VAEs use `VAELoader`

The pinned REDGraft graph loads both the video and audio VAE through Comfy core
`VAELoader`. DuckMotion does the same. `LTXVAudioVAELoader` searches the
checkpoint category and is not the loader contract used by this recipe.

## Validation

Unit/contract coverage lives in `tests/test_ltx_convrot.py` and protects:

- exact high/low sigma schedules;
- guide/preprocess/upscale/decode/output constants;
- independent reproducible stage-one/stage-two noise streams;
- classic Comfy `FUNCTION` and modern class/instance `execute` invocation;
- required pinned-Comfy node readiness;
- asset discovery across a normal model-root layout;
- private ConvRot routing and public architecture-free model payloads;
- the critical REDGraft AV and stage-two operation ordering.

`tools/run_hardware_smoke.py` adds dedicated ConvRot T2V/I2V canaries and a
heavier saved-default row. It intentionally keeps ConvRot distinct from the
standard Diffusers LTX target.

A passing test suite is **not** a hardware validation. Before declaring the
runtime production-ready on a specific GPU, run a real T2V and I2V generation
with the actual REDGraft/support weights and confirm synchronized audio/video,
output dimensions, frame count, memory behavior, and successful teardown.
