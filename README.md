# DuckMotion

DuckMotion is a separately managed WebbDuck web plugin for Wan2.2 video workflows.

Current status:

- Phase 1: setup/readiness checks, image staging, and a raw ComfyUI prompt bridge
- Phase 2 (planned): guided one-click Wan I2V workflow submission

## What It Does (Phase 1)

- Detects Wan2.2 I2V GGUF high/low pairs
- Detects companion files (UMT5 text encoder + Wan VAE)
- Checks ComfyUI connectivity
- Stages images (upload or copy from WebbDuck outputs)
- Submits raw ComfyUI `/prompt` JSON and tracks prompt IDs / history outputs

## Repo Layout

```text
DuckMotion/
|- plugin.json
|- backend.py
`- ui/
   |- index.html
   |- app.js
   `- styles.css
```

## Install Into WebbDuck

WebbDuck discovers web plugins under:

```text
<plugins-root>/webapps/<plugin-id>/
```

Since this repo is standalone, install it by symlinking or copying it into a `webapps/duckmotion` folder.

Example layout:

```text
<webbduck-plugin-root>/
`- webapps/
   `- duckmotion/   -> (symlink or copy from this repo)
```

Plugin roots are searched in WebbDuck using:

1. `WEBBDUCK_PLUGINS_DIR`
2. `webbduck/plugins/`
3. `~/.webbduck/plugins/`

## Configuration

DuckMotion stores plugin state under:

```text
~/.webbduck/plugin_state/
```

Optional environment variables:

- `DUCKMOTION_COMFY_URL`
- `DUCKMOTION_MODELS_DIR`
- `DUCKMOTION_COMFY_MODELS_DIR`

## Setup Notes

- Configure a ComfyUI URL (example: `127.0.0.1:8188`)
- Point `Wan GGUF Folder` to your Wan2.2 I2V GGUF files
- Optionally point `ComfyUI Models Folder` to help companion-file detection

## Development

This plugin imports WebbDuck runtime modules at execution time (for output access and staging integration), so test it in a WebbDuck environment.
