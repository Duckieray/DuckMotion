from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
for candidate in (ROOT, PARENT):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

import backend as duckmotion_backend


def test_probe_diffusers_support_marks_gguf_unavailable_without_gguf_package(monkeypatch):
    fake_diffusers = SimpleNamespace(
        __version__="0.36.0",
        WanImageToVideoPipeline=type("WanImageToVideoPipeline", (), {}),
        GGUFQuantizationConfig=type("GGUFQuantizationConfig", (), {}),
        WanTransformer3DModel=type("WanTransformer3DModel", (), {"from_single_file": staticmethod(lambda *_a, **_k: None)}),
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.delitem(sys.modules, "gguf", raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "gguf":
            raise ModuleNotFoundError("No module named 'gguf'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = duckmotion_backend._probe_diffusers_support()

    assert result["ready"] is True
    assert result["gguf_ready"] is False
    assert "gguf" in result["gguf_error"].lower()


class _FakePipe:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object | None]] = []

    def enable_attention_slicing(self, mode: object = None) -> None:
        self.calls.append(("enable_attention_slicing", mode))

    def enable_vae_slicing(self) -> None:
        self.calls.append(("enable_vae_slicing", None))

    def enable_vae_tiling(self) -> None:
        self.calls.append(("enable_vae_tiling", None))

    def enable_model_cpu_offload(self) -> None:
        self.calls.append(("enable_model_cpu_offload", None))

    def enable_sequential_cpu_offload(self) -> None:
        self.calls.append(("enable_sequential_cpu_offload", None))

    def enable_group_offload(self, **kwargs) -> None:
        self.calls.append(("enable_group_offload", kwargs))


def test_auto_policy_uses_aggressive_low_vram_strategy(monkeypatch):
    monkeypatch.delenv("DUCKMOTION_MEMORY_POLICY", raising=False)
    strategy = duckmotion_backend._plan_memory_strategy(
        {},
        {"device": "cuda", "dtype": "float16", "total_vram_gb": 8.0},
    )

    assert strategy["requested_policy"] == "auto"
    assert strategy["effective_policy"] == "aggressive"
    assert strategy["offload_strategy"] == "sequential_cpu_offload"
    assert strategy["attention_slicing"] == "max"
    assert strategy["vae_slicing"] is True
    assert strategy["vae_tiling"] is True


def test_auto_policy_disables_extra_reductions_on_high_vram_full_cuda(monkeypatch):
    monkeypatch.delenv("DUCKMOTION_MEMORY_POLICY", raising=False)
    strategy = duckmotion_backend._plan_memory_strategy(
        {"cuda_mode": "full"},
        {"device": "cuda", "dtype": "float16", "total_vram_gb": 24.0},
    )

    assert strategy["effective_policy"] == "off"
    assert strategy["offload_strategy"] == "none"
    assert strategy["attention_slicing"] is None
    assert strategy["vae_slicing"] is False
    assert strategy["vae_tiling"] is False


def test_effective_cuda_mode_prefers_offload_for_16gb_gguf_windows(monkeypatch):
    monkeypatch.delenv("DUCKMOTION_CUDA_MODE", raising=False)
    monkeypatch.setattr(duckmotion_backend.os, "name", "nt")

    mode = duckmotion_backend._resolve_effective_cuda_mode(
        {},
        {"device": "cuda", "total_vram_gb": 15.92},
        "diffusers_gguf",
    )

    assert mode == "offload"


def test_effective_cuda_mode_prefers_full_for_high_vram_gguf(monkeypatch):
    monkeypatch.delenv("DUCKMOTION_CUDA_MODE", raising=False)
    monkeypatch.setattr(duckmotion_backend.os, "name", "nt")

    mode = duckmotion_backend._resolve_effective_cuda_mode(
        {},
        {"device": "cuda", "total_vram_gb": 24.0},
        "diffusers_gguf",
    )

    assert mode == "full"


def test_effective_offload_strategy_prefers_group_offload_for_mid_vram_gguf():
    strategy = duckmotion_backend._resolve_effective_offload_strategy(
        {"offload_strategy": "sequential_cpu_offload"},
        {"device": "cuda", "total_vram_gb": 15.92},
        "diffusers_gguf",
    )

    assert strategy == "group_offload"


def test_effective_offload_strategy_keeps_sequential_for_low_vram_gguf():
    strategy = duckmotion_backend._resolve_effective_offload_strategy(
        {"offload_strategy": "sequential_cpu_offload"},
        {"device": "cuda", "total_vram_gb": 8.0},
        "diffusers_gguf",
    )

    assert strategy == "sequential_cpu_offload"


def test_apply_memory_optimizations_gracefully_skips_missing_methods():
    pipe = _FakePipe()
    strategy = {
        "requested_policy": "aggressive",
        "effective_policy": "aggressive",
        "offload_strategy": "sequential_cpu_offload",
        "attention_slicing": "max",
        "vae_slicing": True,
        "vae_tiling": True,
    }

    duckmotion_backend._apply_pipeline_memory_optimizations(pipe, strategy)

    assert strategy["applied_optimizations"] == [
        "attention_slicing=max",
        "enable_vae_slicing",
        "enable_vae_tiling",
    ]
    assert strategy["skipped_optimizations"] == []


def test_enable_cuda_offload_falls_back_to_model_offload():
    class _ModelOnlyPipe:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def enable_model_cpu_offload(self) -> None:
            self.calls.append("enable_model_cpu_offload")

    pipe = _ModelOnlyPipe()
    strategy = {"offload_strategy": "sequential_cpu_offload"}

    duckmotion_backend._enable_cuda_offload(pipe, strategy)

    assert strategy["offload_strategy"] == "model_cpu_offload"
    assert pipe.calls == ["enable_model_cpu_offload"]


def test_enable_cuda_offload_uses_group_offload_when_requested():
    pipe = _FakePipe()
    strategy = {"offload_strategy": "group_offload"}

    duckmotion_backend._enable_cuda_offload(pipe, strategy)

    assert strategy["offload_strategy"] == "group_offload"
    assert pipe.calls[0][0] == "enable_group_offload"


def test_prepare_runtime_for_wan_unloads_webbduck_models(monkeypatch):
    calls: list[str] = []

    class _FakePipelineManager:
        def unload_all(self) -> None:
            calls.append("unload_all")

    def _fake_unload_captioners() -> None:
        calls.append("unload_captioners")

    monkeypatch.setattr(
        duckmotion_backend,
        "_get_webbduck_cleanup_hooks",
        lambda: (_FakePipelineManager(), _fake_unload_captioners),
    )
    monkeypatch.setattr(duckmotion_backend, "_cleanup_torch_memory", lambda: calls.append("cleanup_torch_memory"))

    duckmotion_backend._prepare_runtime_for_wan({"device": "cuda"})

    assert calls == ["unload_all", "unload_captioners", "cleanup_torch_memory"]


def test_evaluate_job_safety_blocks_risky_windows_job(monkeypatch):
    monkeypatch.setattr(duckmotion_backend.os, "name", "nt")
    monkeypatch.setattr(
        duckmotion_backend,
        "_get_host_memory_snapshot",
        lambda: {"os": "nt", "avail_phys_gb": 6.0, "avail_pagefile_gb": 8.0, "configured_pagefile_gb": 7.8},
    )

    result = duckmotion_backend._evaluate_job_safety(
        {"width": 832, "height": 480, "num_frames": 81, "num_inference_steps": 30},
        {"cuda_mode": "full", "memory_policy": "aggressive", "safety_mode": "block"},
        {"device": "cuda", "dtype": "float16", "total_vram_gb": 12.0},
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["risk_level"] == "high"
    assert result["reasons"]
    assert "blocked unsafe duckmotion wan job" in result["message"].lower()


def test_evaluate_job_safety_allows_safer_windows_job(monkeypatch):
    monkeypatch.setattr(duckmotion_backend.os, "name", "nt")
    monkeypatch.setattr(
        duckmotion_backend,
        "_get_host_memory_snapshot",
        lambda: {"os": "nt", "avail_phys_gb": 20.0, "avail_pagefile_gb": 40.0, "configured_pagefile_gb": 64.0},
    )

    result = duckmotion_backend._evaluate_job_safety(
        {"width": 576, "height": 320, "num_frames": 49, "num_inference_steps": 20},
        {"cuda_mode": "offload", "memory_policy": "aggressive", "safety_mode": "block"},
        {"device": "cuda", "dtype": "float16", "total_vram_gb": 12.0},
    )

    assert result["ok"] is True
    assert result["blocked"] is False
    assert result["reasons"] == []


def test_evaluate_job_safety_blocks_a14b_load_risk_even_for_smaller_job(monkeypatch):
    monkeypatch.setattr(duckmotion_backend.os, "name", "nt")
    monkeypatch.setattr(
        duckmotion_backend,
        "_get_host_memory_snapshot",
        lambda: {"os": "nt", "avail_phys_gb": 18.0, "avail_pagefile_gb": 30.0, "configured_pagefile_gb": 7.8},
    )

    result = duckmotion_backend._evaluate_job_safety(
        {"width": 576, "height": 320, "num_frames": 49, "num_inference_steps": 20},
        {
            "cuda_mode": "offload",
            "memory_policy": "aggressive",
            "safety_mode": "block",
            "model_id_or_path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        },
        {"device": "cuda", "dtype": "float16", "total_vram_gb": 12.0},
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert any("a14b" in reason.lower() for reason in result["reasons"])


def test_evaluate_job_safety_allows_explicit_gguf_selection_even_if_probe_plan_would_fall_back(monkeypatch):
    monkeypatch.setattr(duckmotion_backend.os, "name", "nt")
    monkeypatch.setattr(
        duckmotion_backend,
        "_get_host_memory_snapshot",
        lambda: {"os": "nt", "avail_phys_gb": 18.0, "avail_pagefile_gb": 30.0, "configured_pagefile_gb": 7.8},
    )
    monkeypatch.setattr(
        duckmotion_backend,
        "_discover_local_models",
        lambda _config: {"gguf_items": [], "gguf_candidates": []},
    )

    result = duckmotion_backend._evaluate_job_safety(
        {"width": 576, "height": 320, "num_frames": 49, "num_inference_steps": 20},
        {
            "cuda_mode": "offload",
            "memory_policy": "aggressive",
            "safety_mode": "block",
            "model_id_or_path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "runtime_backend": "diffusers_gguf",
            "gguf_transformer_path": "C:/models/wan/wan-q8.gguf",
        },
        {"device": "cuda", "dtype": "float16", "total_vram_gb": 12.0},
    )

    assert result["ok"] is True
    assert result["blocked"] is False
    assert result["reasons"] == []


def test_runtime_plan_prefers_gguf_for_lower_vram_cuda():
    gguf_items = [{"path": "C:/models/wan/wan-q4.gguf", "label": "wan-q4.gguf"}]
    gguf_candidates = [{"label": "wan-q4", "paths": {"single": "C:/models/wan/wan-q4.gguf"}, "complete": True}]

    plan = duckmotion_backend._plan_runtime_backend(
        {"runtime_backend": "auto", "model_id_or_path": "C:/models/wan/wan-q4.gguf"},
        {"device": "cuda", "total_vram_gb": 12.0},
        diffusers_status={"ready": True, "gguf_ready": True},
        discovered_gguf_items=gguf_items,
        gguf_candidates=gguf_candidates,
    )

    assert plan["selected_backend"] == "diffusers_gguf"
    assert plan["gguf_transformer_path"] == "C:/models/wan/wan-q4.gguf"


def test_runtime_plan_stays_diffusers_without_configured_gguf_selection():
    gguf_items = [{"path": "C:/models/wan/wan-q4.gguf", "label": "wan-q4.gguf"}]
    gguf_candidates = [{"label": "wan-q4", "paths": {"single": "C:/models/wan/wan-q4.gguf"}, "complete": True}]

    plan = duckmotion_backend._plan_runtime_backend(
        {"runtime_backend": "auto", "model_id_or_path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers"},
        {"device": "cuda", "total_vram_gb": 12.0},
        diffusers_status={"ready": True, "gguf_ready": True},
        discovered_gguf_items=gguf_items,
        gguf_candidates=gguf_candidates,
    )

    assert plan["selected_backend"] == "diffusers"
    assert plan["gguf_transformer_path"] is None


def test_runtime_plan_falls_back_when_gguf_unavailable():
    plan = duckmotion_backend._plan_runtime_backend(
        {"runtime_backend": "diffusers_gguf"},
        {"device": "cuda", "total_vram_gb": 12.0},
        diffusers_status={"ready": True, "gguf_ready": False},
        discovered_gguf_items=[],
    )

    assert plan["selected_backend"] == "diffusers"
    assert "falling back" in plan["reason"]


def test_group_gguf_candidates_pairs_h_and_l_models():
    items = [
        duckmotion_backend._discovered_gguf_item(path=Path("C:/wan/wanModel_Q8H.gguf"), source="configured_models_dir"),
        duckmotion_backend._discovered_gguf_item(path=Path("C:/wan/wanModel_Q8L.gguf"), source="configured_models_dir"),
    ]

    grouped = duckmotion_backend._group_gguf_candidates(items)

    assert len(grouped) == 1
    assert grouped[0]["complete"] is True
    assert grouped[0]["paths"]["H"].endswith("wanModel_Q8H.gguf")
    assert grouped[0]["paths"]["L"].endswith("wanModel_Q8L.gguf")


def test_resolve_gguf_transformer_path_prefers_pair_high_variant():
    path = duckmotion_backend._resolve_gguf_transformer_path(
        {"runtime_backend": "diffusers_gguf"},
        [],
        [{"paths": {"H": "C:/wan/wanModel_Q8H.gguf", "L": "C:/wan/wanModel_Q8L.gguf"}, "complete": True}],
    )

    assert path == "C:/wan/wanModel_Q8H.gguf"


def test_resolve_gguf_transformer_path_prefers_configured_path():
    path = duckmotion_backend._resolve_gguf_transformer_path(
        {"gguf_transformer_path": "C:/custom/wan.gguf"},
        [{"path": "C:/auto/wan.gguf"}],
    )

    assert path == "C:/custom/wan.gguf"


def test_resolve_gguf_transformer_path_uses_model_source_when_quantized_path_selected():
    path = duckmotion_backend._resolve_gguf_transformer_path(
        {"model_id_or_path": "C:/custom/wan-q8.gguf", "runtime_backend": "auto"},
        [],
        [],
    )

    assert path == "C:/custom/wan-q8.gguf"


def test_resolve_local_diffusers_source_prefers_matching_hf_cache_snapshot():
    resolved = duckmotion_backend._resolve_local_diffusers_source(
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        [
            {
                "repo_id": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "path": "C:/hf-cache/models--Wan-AI--Wan2.2-I2V-A14B-Diffusers/snapshots/123",
            }
        ],
    )

    assert resolved == "C:/hf-cache/models--Wan-AI--Wan2.2-I2V-A14B-Diffusers/snapshots/123"


def test_current_setup_compatibility_marks_a14b_incompatible(monkeypatch):
    monkeypatch.setattr(duckmotion_backend.os, "name", "nt")
    monkeypatch.setattr(
        duckmotion_backend,
        "_get_host_memory_snapshot",
        lambda: {"os": "nt", "avail_phys_gb": 18.0, "avail_pagefile_gb": 30.0, "configured_pagefile_gb": 7.8},
    )
    monkeypatch.setattr(
        duckmotion_backend,
        "_probe_diffusers_support",
        lambda: {"ready": True, "gguf_ready": False, "version": "0.36.0"},
    )
    monkeypatch.setattr(duckmotion_backend, "_discover_local_models", lambda _config: {"gguf_items": []})

    result = duckmotion_backend._get_current_setup_compatibility(
        {
            "model_id_or_path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "default_width": 832,
            "default_height": 480,
            "default_frames": 81,
            "default_steps": 30,
            "runtime_backend": "auto",
        },
        {"device": "cuda", "dtype": "float16", "total_vram_gb": 12.0},
    )

    assert result["compatible"] is False
    assert "incompatible" in result["reason"].lower() or "unsafe" in result["reason"].lower()


def test_current_setup_compatibility_marks_safe_profile_compatible(monkeypatch):
    monkeypatch.setattr(duckmotion_backend.os, "name", "nt")
    monkeypatch.setattr(
        duckmotion_backend,
        "_get_host_memory_snapshot",
        lambda: {"os": "nt", "avail_phys_gb": 24.0, "avail_pagefile_gb": 48.0, "configured_pagefile_gb": 64.0},
    )
    monkeypatch.setattr(
        duckmotion_backend,
        "_probe_diffusers_support",
        lambda: {"ready": True, "gguf_ready": False, "version": "0.36.0"},
    )
    monkeypatch.setattr(duckmotion_backend, "_discover_local_models", lambda _config: {"gguf_items": []})

    result = duckmotion_backend._get_current_setup_compatibility(
        {
            "model_id_or_path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "default_width": 576,
            "default_height": 320,
            "default_frames": 49,
            "default_steps": 20,
            "runtime_backend": "auto",
            "cuda_mode": "offload",
            "memory_policy": "aggressive",
        },
        {"device": "cuda", "dtype": "float16", "total_vram_gb": 24.0},
    )

    assert result["compatible"] is True


def test_webbduck_models_dir_env_var_overrides_registry_root(monkeypatch, tmp_path):
    external_models = tmp_path / "external_models"
    external_models.mkdir()

    monkeypatch.setenv("WEBBDUCK_MODELS_DIR", str(external_models))

    import importlib
    import webbduck.models.registry as reg  # noqa: F401

    with monkeypatch.context() as mp:
        mp.setattr(reg, "ROOT", tmp_path / "webbduck_repo", raising=False)
        mp.setattr(reg, "MODELS_ROOT", tmp_path / "webbduck_repo", raising=False)
        mp.setattr(reg, "CHECKPOINT_ROOT", tmp_path / "webbduck_repo" / "checkpoint" / "sdxl", raising=False)
        mp.setattr(reg, "HF_CACHE", tmp_path / ".cache" / "huggingface" / "hub", raising=False)

        hints = duckmotion_backend._webbduck_path_hints()

    assert hints["models_root"] == external_models.resolve()
