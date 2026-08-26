"""Shared WebbDuck host/runtime integration for DuckMotion.

This module is architecture neutral. Video backends only receive normalized
runtime and lease services through the generic coordinator; no model family owns
WebbDuck integration anymore.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


def _ensure_webbduck_root() -> None:
    candidates: list[Path] = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    for raw in list(sys.path):
        try:
            candidate = Path(raw).resolve() if raw else Path.cwd().resolve()
        except Exception:
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if (candidate / "server" / "app.py").exists() and (candidate / "core").is_dir():
            value = str(candidate)
            if value not in sys.path:
                sys.path.insert(0, value)
            return


def runtime_profile_safe() -> dict[str, Any]:
    _ensure_webbduck_root()
    try:
        from core.runtime import resolve_runtime_profile

        profile = resolve_runtime_profile()
        return {
            "ok": True,
            "profile": profile.to_dict(),
            "source": "core.runtime.resolve_runtime_profile",
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def runtime_profile_or_raise() -> dict[str, Any]:
    runtime = runtime_profile_safe()
    if not runtime.get("ok"):
        raise RuntimeError(f"Runtime profile unavailable: {runtime.get('error')}")
    profile = runtime.get("profile")
    if not isinstance(profile, dict):
        raise RuntimeError("Runtime profile returned an invalid payload")
    return dict(profile)


def gpu_lease() -> dict[str, Any]:
    _ensure_webbduck_root()
    try:
        from core.gpu_lease import get_gpu_lease

        value = get_gpu_lease()
        return dict(value) if isinstance(value, dict) else {"held": False}
    except Exception:
        return {"held": False}


def acquire_gpu_lease(**kwargs: Any) -> dict[str, Any]:
    _ensure_webbduck_root()
    try:
        from core.gpu_lease import acquire_gpu_lease_blocking
    except Exception as exc:
        raise RuntimeError(f"WebbDuck GPU lease service unavailable: {exc}") from exc
    value = acquire_gpu_lease_blocking(**kwargs)
    return dict(value) if isinstance(value, dict) else {"lease": None}


def release_gpu_lease(**kwargs: Any) -> Any:
    _ensure_webbduck_root()
    try:
        from core.gpu_lease import release_gpu_lease as release
    except Exception as exc:
        raise RuntimeError(f"WebbDuck GPU lease service unavailable: {exc}") from exc
    return release(**kwargs)
