# LTX I2V reference stability

DuckMotion's LTX-2.5 ConvRot runtime separates three concerns:

1. the checkpoint companion owns sampler/sigma/CFG tuning;
2. the execution profile owns the two-stage AV topology and support assets;
3. an optional I2V stability mode controls how strongly the source image is used as a temporal reference.

The browser never selects a backend or model family. A model profile advertises `i2v_stability_modes` through its public constraints; the UI shows this control only when the selected model supports it.

## Reference conditioning

The LTX-2.5 ConvRot I2V worker uses Comfy core's `LTXVAddGuide` rather than the older `LTXVImgToVideoInplace` path.

`LTXVAddGuide` does more than replace frame-zero latent pixels. It appends reference/keyframe tokens and records their temporal positions in positive and negative conditioning, allowing the sampler's attention path to retain the reference throughout the generated sequence. `LTXVCropGuides` removes those temporary guide tokens before latent upscaling/final decode, and the references are rebuilt at the high-resolution stage.

DuckMotion invokes these known Comfy nodes directly. It does not execute arbitrary companion workflow JSON.

## Modes

### Model default

Preserves the normalized companion recipe's source-image guide strengths and stage-two noise policy. It still uses the current `LTXVAddGuide` reference-conditioning primitive.

Use this for general-purpose I2V and action-heavy shots.

### Identity stable

Keeps only a first-frame reference, but applies a quality-oriented stability override:

- first-stage source reference: `1.0`
- high-resolution source reference: `1.0`
- stage-two noise policy: `same_seed`

This is intended for clips where subject/face identity matters more than maximizing freedom in the refinement pass.

### Locked shot

Includes all `Identity stable` behavior and uses the same source image as both:

- frame `0` reference; and
- final-frame (`-1`) reference.

This is intended for tripod shots, portraits, interviews, product shots, and other low-motion compositions where the first-frame appearance should remain recognizable over a long generation.

The last-frame reference is not a duplicated output frame. It is an LTX temporal/keyframe reference that constrains the generated sequence between the two anchors.

## Metadata

Every I2V result records both the checkpoint-author recipe and DuckMotion's effective stability layer:

```text
companion_execution_recipe
execution_recipe
i2v_stability_mode
i2v_stability_overrides
reference_conditioning
stage1_guide_plan
stage2_guide_plan
stage2_seed
```

This keeps visual A/B tests reproducible and makes quality overrides explicit rather than silently rewriting checkpoint-author tuning.

## Deliberately separate concerns

Reference stability does not change:

- sampler or custom sigma schedules;
- CFG unless the companion recipe does;
- model/VAE/upscaler selection;
- DynamicVRAM/offload policy;
- frame rate;
- requested duration/frame count;
- current dimension snapping.

Frame interpolation and native LTX-2.5 sizing are separate quality features and should be evaluated only after the base generated frames are temporally coherent.
