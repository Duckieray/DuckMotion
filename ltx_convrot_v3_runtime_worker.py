"""Embedded-Comfy V3 compatibility launcher for LTX ConvRot.

DuckMotion intentionally calls pinned Comfy nodes directly instead of starting a
Comfy server or executing arbitrary workflow JSON. Modern Comfy V3 nodes expect
the executor to clone their class and attach a HiddenHolder before calling
``execute``. Without that binding, output nodes such as SaveVideo see
``cls.hidden is None`` and fail when reading hidden prompt/metadata fields.

This launcher installs the missing V3 invocation adapter, then delegates to the
existing ConvRot runtime launcher which owns VRAM policy and the full inference-
mode lifetime.
"""

from __future__ import annotations

import asyncio
import inspect

import ltx_convrot_runtime_worker as runtime_worker
import ltx_convrot_worker as recipe_worker


def _install_v3_node_dispatch() -> None:
    """Mirror Comfy's V3 class-clone/hidden-input binding for direct node calls."""

    original_call_node = recipe_worker._call_node
    if getattr(original_call_node, "_duckmotion_v3_dispatch", False):
        return

    def call_node(nodes_module, node_name: str, **kwargs):
        cls = nodes_module.NODE_CLASS_MAPPINGS.get(node_name)
        if cls is None:
            return original_call_node(nodes_module, node_name, **kwargs)

        try:
            from comfy_api.internal import _ComfyNodeInternal, make_locked_method_func
        except Exception:
            return original_call_node(nodes_module, node_name, **kwargs)

        is_v3 = isinstance(cls, type) and issubclass(cls, _ComfyNodeInternal)
        if not is_v3:
            return original_call_node(nodes_module, node_name, **kwargs)

        try:
            # This is the same class preparation used by Comfy's PromptExecutor.
            # No graph metadata exists in DuckMotion's direct invocation path, so
            # PREPARE_CLASS_CLONE(None) creates a HiddenHolder whose fields are
            # safely None. V3 nodes can then use their normal hidden-input API.
            cls.VALIDATE_CLASS()
            class_clone = cls.PREPARE_CLASS_CLONE(None)
            function = make_locked_method_func(cls, "execute", class_clone)

            signature = inspect.signature(function)
            has_var_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            call_kwargs = (
                kwargs
                if has_var_kwargs
                else {
                    key: value
                    for key, value in kwargs.items()
                    if key in signature.parameters
                }
            )
            value = function(**call_kwargs)
            if inspect.isawaitable(value):
                value = asyncio.run(value)
            return recipe_worker._result_tuple(value)
        except Exception as exc:
            raise RuntimeError(f"Comfy node {node_name!r} failed: {exc}") from exc

    call_node._duckmotion_v3_dispatch = True
    recipe_worker._call_node = call_node


def main() -> int:
    _install_v3_node_dispatch()
    return runtime_worker.main()


if __name__ == "__main__":
    raise SystemExit(main())
