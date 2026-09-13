"""Read optional WebbDuck model-provider credentials without depending on WebbDuck.

DuckMotion setup works anonymously by default.  When a user has chosen to save
provider credentials in WebbDuck Settings, this module understands the shared
local file format so CLI setup can use those credentials without requiring the
WebbDuck server to be running or shell exports to be configured.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


_PROVIDER_ENV = {
    "huggingface": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
    "civitai": ("CIVITAI_TOKEN", "CIVITAI_API_TOKEN", "CIVITAI_API_KEY"),
}
_CANONICAL_ENV = {
    "huggingface": "HF_TOKEN",
    "civitai": "CIVITAI_TOKEN",
}


def credentials_path() -> Path:
    raw = str(os.getenv("WEBBDUCK_CREDENTIALS_FILE") or "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".webbduck" / "provider_credentials.json"


def _stored_token(provider: str) -> str:
    path = credentials_path()
    if not path.exists() or not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return ""
    providers = payload.get("providers") if isinstance(payload.get("providers"), dict) else {}
    entry = providers.get(provider) if isinstance(providers.get(provider), dict) else {}
    return str(entry.get("token") or "").strip()


def provider_token(provider: str) -> str:
    provider = str(provider or "").strip().lower()
    env_names = _PROVIDER_ENV.get(provider)
    if env_names is None:
        raise ValueError(f"Unknown provider: {provider}")
    for env_name in env_names:
        token = str(os.getenv(env_name) or "").strip()
        if token:
            return token
    return _stored_token(provider)


def apply_provider_credentials_to_environment() -> None:
    """Populate canonical variables only when the user did not override them."""
    for provider, canonical in _CANONICAL_ENV.items():
        if any(str(os.getenv(name) or "").strip() for name in _PROVIDER_ENV[provider]):
            continue
        token = _stored_token(provider)
        if token:
            os.environ[canonical] = token
