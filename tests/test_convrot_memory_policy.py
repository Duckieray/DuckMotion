from __future__ import annotations

import inspect

import ltx_convrot_runtime_worker
from ltx_convrot_runtime_worker import resolve_memory_policy
from model_recipes import LTX25_CONVROT_TWO_STAGE_AV


def test_convrot_auto_uses_conservative_policy_on_16gb_class_gpu(monkeypatch):
    monkeypatch.delenv("DUCKMOTION_LTX_CONVROT_MEMORY", raising=False)

    policy = resolve_memory_policy(15.51)

    assert policy["name"] == "conservative"
    args = policy["comfy_args"]
    assert "--enable-dynamic-vram" in args
    assert "--cache-none" in args
    assert "--disable-smart-memory" in args
    assert "--use-split-cross-attention" in args
    assert args[args.index("--vram-headroom") + 1] == "1.5"


def test_convrot_auto_uses_balanced_policy_on_24gb_gpu(monkeypatch):
    monkeypatch.delenv("DUCKMOTION_LTX_CONVROT_MEMORY", raising=False)

    policy = resolve_memory_policy(24.0)

    assert policy["name"] == "balanced"
    args = policy["comfy_args"]
    assert "--enable-dynamic-vram" in args
    assert "--cache-none" in args
    assert "--disable-smart-memory" not in args
    assert "--use-split-cross-attention" not in args
    assert args[args.index("--vram-headroom") + 1] == "0.75"


def test_convrot_auto_keeps_large_gpu_on_performance_policy(monkeypatch):
    monkeypatch.delenv("DUCKMOTION_LTX_CONVROT_MEMORY", raising=False)

    policy = resolve_memory_policy(48.0)

    assert policy["name"] == "performance"
    assert policy["comfy_args"] == ("--disable-dynamic-vram", "--cache-none")


def test_convrot_memory_policy_can_be_overridden_for_debugging(monkeypatch):
    monkeypatch.setenv("DUCKMOTION_LTX_CONVROT_MEMORY", "balanced")

    policy = resolve_memory_policy(15.51)

    assert policy["name"] == "balanced"


def test_execution_profile_uses_vram_aware_runtime_launcher():
    assert LTX25_CONVROT_TWO_STAGE_AV.worker == "ltx_convrot_runtime_worker.py"


def test_embedded_comfy_enables_cli_parsing_before_memory_modules():
    source = inspect.getsource(ltx_convrot_runtime_worker._initialize_aimdo_control)
    assert "comfy.options.enable_args_parsing()" in source
    assert source.index("comfy.options.enable_args_parsing()") < source.index("from comfy.cli_args import args")
    assert "aimdo_control.init(" in source


def test_embedded_comfy_activates_dynamic_model_patcher():
    source = inspect.getsource(ltx_convrot_runtime_worker._activate_dynamic_vram)
    assert "aimdo_control.init_devices(" in source
    assert "model_patcher.CoreModelPatcher = model_patcher.ModelPatcherDynamic" in source
    assert "memory_management.aimdo_enabled = True" in source


def test_required_dynamic_policy_fails_closed_if_activation_is_missing():
    source = inspect.getsource(ltx_convrot_runtime_worker._prepare_comfy)
    assert 'if policy["dynamic_vram"] and not dynamic_vram_active:' in source
    assert "requires Comfy DynamicVRAM" in source


def test_embedded_comfy_full_recipe_run_stays_inside_inference_mode(monkeypatch):
    import torch

    observed = []

    def fake_run(request, output_dir):
        observed.append(torch.is_inference_mode_enabled())
        # Simulate work at two separate points in the recipe lifetime. The outer
        # wrapper must remain active continuously instead of entering/exiting per
        # node dispatch.
        tensor = torch.ones(1)
        tensor.add_(1)
        observed.append(torch.is_inference_mode_enabled())
        return {"ok": True, "value": int(tensor.item())}

    monkeypatch.setattr(
        ltx_convrot_runtime_worker.recipe_worker,
        "_run",
        fake_run,
    )

    ltx_convrot_runtime_worker._install_full_inference_run_context()
    result = ltx_convrot_runtime_worker.recipe_worker._run({}, None)

    assert result == {"ok": True, "value": 2}
    assert observed == [True, True]


def test_full_inference_context_is_installed_by_runtime_main():
    source = inspect.getsource(ltx_convrot_runtime_worker.main)
    assert "_install_full_inference_run_context()" in source
    assert "_install_inference_node_dispatch" not in source
