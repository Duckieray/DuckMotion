"""Shared runtime path resolution for DuckMotion backends and tools.

Normal users should not need to export backend-specific interpreter variables.
Runtime preparation owns a deterministic directory under the user's data home;
environment variables remain advanced overrides only.
"""

from __future__ import annotations

import os
from pathlib import Path


RUNTIME_ENV_VARS = {
    "wan": "DUCKMOTION_WAN_PYTHON",
    "ltx25": "DUCKMOTION_LTX_PYTHON",
    "ltx25_convrot": "DUCKMOTION_LTX_CONVROT_PYTHON",
}
CONVROT_COMFY_ROOT_ENV = "DUCKMOTION_LTX_CONVROT_COMFY_ROOT"


def runtime_home() -> Path:
    return Path(
        os.getenv("DUCKMOTION_RUNTIME_HOME", "~/.local/share/duckmotion/runtimes")
    ).expanduser()


def python_in_runtime(runtime_root: Path) -> Path:
    windows = runtime_root / "Scripts" / "python.exe"
    if windows.exists():
        return windows
    return runtime_root / "bin" / "python"


def default_runtime_python(runtime: str) -> Path:
    if runtime not in RUNTIME_ENV_VARS:
        raise KeyError(f"Unknown DuckMotion runtime: {runtime}")
    return python_in_runtime(runtime_home() / runtime)


def resolve_runtime_python(runtime: str) -> str:
    """Return an explicit override or DuckMotion's deterministic runtime path."""
    env_var = RUNTIME_ENV_VARS[runtime]
    override = str(os.getenv(env_var) or "").strip()
    if override:
        return str(Path(override).expanduser())
    return str(default_runtime_python(runtime))


def runtime_root_from_python(python_exe: str | Path) -> Path:
    """Return the owning venv/runtime directory without resolving interpreter symlinks.

    On POSIX, ``venv/bin/python`` is commonly a symlink to the base interpreter.
    Calling ``Path.resolve()`` here would escape the DuckMotion runtime and make
    sibling runtime-owned resources (for example the pinned ConvRot Comfy checkout)
    appear to be missing even though setup installed them correctly.
    """
    path = Path(python_exe).expanduser()
    parent = path.parent
    if parent.name.lower() in {"bin", "scripts"}:
        return parent.parent
    return parent


def resolve_convrot_comfy_root() -> Path:
    """Return an explicit ConvRot Comfy override or its runtime-owned default."""
    override = str(os.getenv(CONVROT_COMFY_ROOT_ENV) or "").strip()
    if override:
        return Path(override).expanduser()
    return runtime_root_from_python(resolve_runtime_python("ltx25_convrot")) / "comfyui"


def configure_default_runtime_env() -> dict[str, str]:
    """Populate backend env vars only when the user has not overridden them.

    Existing backend modules already use these variables as advanced overrides.
    Setting deterministic defaults once at plugin startup keeps that contract
    while removing environment-variable setup from the normal installation path.
    Runtime-owned adjunct paths are derived from the *venv path itself*, never
    from the symlink-resolved base interpreter.
    """
    configured: dict[str, str] = {}
    for runtime, env_var in RUNTIME_ENV_VARS.items():
        current = str(os.getenv(env_var) or "").strip()
        value = current or str(default_runtime_python(runtime))
        if not current:
            os.environ[env_var] = value
        configured[env_var] = value

    comfy_current = str(os.getenv(CONVROT_COMFY_ROOT_ENV) or "").strip()
    comfy_value = comfy_current or str(resolve_convrot_comfy_root())
    if not comfy_current:
        os.environ[CONVROT_COMFY_ROOT_ENV] = comfy_value
    configured[CONVROT_COMFY_ROOT_ENV] = comfy_value
    return configured


def runtime_status(runtime: str) -> dict[str, object]:
    env_var = RUNTIME_ENV_VARS[runtime]
    override = str(os.getenv(env_var) or "").strip()
    path = Path(resolve_runtime_python(runtime)).expanduser()
    default_path = default_runtime_python(runtime)
    return {
        "runtime": runtime,
        "python": str(path),
        "exists": path.exists() and path.is_file(),
        "source": "environment" if override and path != default_path else "default",
        "override_env": env_var,
    }
