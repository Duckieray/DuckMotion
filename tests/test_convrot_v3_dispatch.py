from __future__ import annotations

import inspect

import ltx_convrot_v3_runtime_worker
from model_recipes import LTX25_CONVROT_TWO_STAGE_AV


def test_convrot_profile_routes_through_v3_runtime_adapter():
    assert LTX25_CONVROT_TWO_STAGE_AV.worker == "ltx_convrot_v3_runtime_worker.py"


def test_v3_dispatch_mirrors_comfy_class_clone_and_hidden_binding():
    source = inspect.getsource(ltx_convrot_v3_runtime_worker._install_v3_node_dispatch)
    assert "issubclass(cls, _ComfyNodeInternal)" in source
    assert "cls.VALIDATE_CLASS()" in source
    assert "cls.PREPARE_CLASS_CLONE(None)" in source
    assert 'make_locked_method_func(cls, "execute", class_clone)' in source
    assert "recipe_worker._result_tuple(value)" in source


def test_v3_dispatch_keeps_non_v3_nodes_on_existing_recipe_dispatch():
    source = inspect.getsource(ltx_convrot_v3_runtime_worker._install_v3_node_dispatch)
    assert "if not is_v3:" in source
    assert "return original_call_node(nodes_module, node_name, **kwargs)" in source


def test_v3_dispatch_adds_node_name_to_runtime_errors():
    source = inspect.getsource(ltx_convrot_v3_runtime_worker._install_v3_node_dispatch)
    assert 'Comfy node {node_name!r} failed' in source
