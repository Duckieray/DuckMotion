from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "duckmotion_hardware_smoke",
    ROOT / "tools" / "run_hardware_smoke.py",
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_model_matching_uses_public_name_or_source():
    items = [
        {"name": "Wan2.2 TI2V", "source": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"},
        {"name": "LTX-2.5", "source": "Lightricks/LTX-2.5-Diffusers"},
    ]
    assert smoke._find_model(items, None, ("wan2.2", "ti2v", "5b"))["source"].startswith("Wan-AI/")
    assert smoke._find_model(items, None, ("ltx-2.5",))["source"].startswith("Lightricks/")


def test_model_identity_prefers_public_source():
    item = {"name": "LTX-2.5", "source": "Lightricks/LTX-2.5-Diffusers"}
    assert smoke._model_identity(item) == "Lightricks/LTX-2.5-Diffusers"


def test_heavy_rows_are_explicitly_marked():
    class Args:
        wan_5b_model = None
        wan_i2v_model = None
        ltx_model = None

    items = [
        {"name": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "source": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"},
        {"name": "Wan-AI/Wan2.2-I2V-A14B-Diffusers", "source": "Wan-AI/Wan2.2-I2V-A14B-Diffusers"},
        {"name": "Lightricks/LTX-2.5-Diffusers", "source": "Lightricks/LTX-2.5-Diffusers"},
    ]
    rows = {row["row"]: row for row in smoke._rows(Args(), items, "/tmp/source.png")}
    assert rows["wan-ti2v-5b-canary"]["heavy"] is False
    assert rows["ltx25-t2v-canary"]["heavy"] is False
    assert rows["wan-ti2v-5b-default"]["heavy"] is True
    assert rows["wan-i2v-a14b-canary"]["heavy"] is True
    assert rows["ltx25-t2v-default"]["heavy"] is True


def test_runner_requires_explicit_generation_and_heavy_opt_in():
    source = (ROOT / "tools" / "run_hardware_smoke.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--execute", action="store_true"' in source
    assert 'parser.add_argument("--include-heavy", action="store_true"' in source
    assert "weights.get(\"present\") is not True" in source
    assert '"/runtime-readiness"' in source
    assert '"/engine/generate"' in source
    assert '"/engine/unload"' in source
