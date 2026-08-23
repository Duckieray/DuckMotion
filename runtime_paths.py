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


def configure_default_runtime_env() -> dict[str, str]:
    """Populate backend env vars only when the user has not overridden them.

    Existing backend modules already use these variables as advanced overrides.
    Setting deterministic defaults once at plugin startup keeps that contract
    while removing environment-variable setup from the normal installation path.
    """
    configured: dict[str, str] = {}
    for runtime, env_var in RUNTIME_ENV_VARS.items():
        current = str(os.getenv(env_var) or "").strip()
        value = current or str(default_runtime_python(runtime))
        if not current:
            os.environ[env_var] = value
        configured[env_var] = value
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
