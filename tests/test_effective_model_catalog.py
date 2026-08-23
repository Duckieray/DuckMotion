from __future__ import annotations

import plugin_backend


def _convrot_catalog():
    return {
        "items": [
            {
                "name": "Unbranded Community LTX Checkpoint",
                "source": "/models/AcmeCinemaLTX25ConvRotQ8.safetensors",
                "location": "local",
            }
        ],
        "count": 1,
        "hf_cache": "/cache/hub",
    }


def _profile_readiness(_descriptor):
    return {
        "ready": True,
        "reason": None,
        "source_format": "int8_convrot",
        "execution_profile": "ltx25_convrot_two_stage_av",
        "profile_defaults": {
            "width": 1152,
            "height": 768,
            "num_frames": 241,
            "fps": 24,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
        },
        "profile_constraints": {
            "dimension_multiple": 64,
            "frame_count_modulo": 8,
            "frame_count_remainder": 1,
            "generation_stages": 2,
            "sampling_schedule_locked": True,
        },
    }


def test_public_catalog_applies_effective_profile_semantics_without_exposing_profile(monkeypatch):
    monkeypatch.setattr(plugin_backend.services, "load_config", lambda: {})
    monkeypatch.setattr(plugin_backend, "discover_video_models", lambda _config: _convrot_catalog())
    monkeypatch.setattr(plugin_backend, "_register_installed_backends", lambda: None)
    monkeypatch.setattr(plugin_backend.backend_resolver, "readiness", _profile_readiness)

    payload = plugin_backend._public_model_catalog()
    model = payload["items"][0]

    assert model["name"] == "Unbranded Community LTX Checkpoint"
    assert model["defaults"]["width"] == 1152
    assert model["defaults"]["height"] == 768
    assert model["defaults"]["num_frames"] == 241
    assert model["constraints"]["generation_stages"] == 2
    assert model["constraints"]["sampling_schedule_locked"] is True
    assert model["ready"] is True

    # The browser receives effective behavior, not private routing concepts.
    assert "runtime" not in model
    assert "weights" not in model
    assert "execution_profile" not in model
    assert "source_format" not in model
    assert "architecture" not in model
    assert "backend" not in model


def test_runtime_readiness_keeps_profile_identity_for_diagnostics(monkeypatch):
    monkeypatch.setattr(plugin_backend.services, "load_config", lambda: {})
    monkeypatch.setattr(plugin_backend, "discover_video_models", lambda _config: _convrot_catalog())
    monkeypatch.setattr(plugin_backend, "_register_installed_backends", lambda: None)
    monkeypatch.setattr(plugin_backend.backend_resolver, "readiness", _profile_readiness)

    payload = plugin_backend._runtime_readiness()
    row = payload["items"][0]

    assert row["defaults"]["width"] == 1152
    assert row["runtime"]["source_format"] == "int8_convrot"
    assert row["runtime"]["execution_profile"] == "ltx25_convrot_two_stage_av"
