import json
from pathlib import Path

from model_discovery import (
    discover_hf_video_models,
    discover_local_video_models,
    discover_video_models,
)


def _model_index(root: Path, class_name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": class_name}),
        encoding="utf-8",
    )


def test_local_discovery_finds_wan_and_ltx_without_engine_grouping(tmp_path):
    wan = tmp_path / "checkpoint" / "wan" / "Wan2.2-I2V-A14B-Diffusers"
    _model_index(wan, "WanImageToVideoPipeline")

    ltx = tmp_path / "checkpoint" / "ltx2.5" / "LTX-2.5-Diffusers"
    _model_index(ltx, "LTX2ImageToVideoPipeline")

    items = discover_local_video_models([tmp_path / "checkpoint"])
    by_name = {item["name"]: item for item in items}

    assert by_name["Wan2.2-I2V-A14B-Diffusers"]["capabilities"]["image_to_video"] is True
    assert by_name["Wan2.2-I2V-A14B-Diffusers"]["supported"] is True
    assert by_name["LTX-2.5-Diffusers"]["capabilities"]["text_to_video"] is True
    assert by_name["LTX-2.5-Diffusers"]["capabilities"]["audio_output"] is True
    assert by_name["LTX-2.5-Diffusers"]["supported"] is False
    assert "architecture" not in by_name["LTX-2.5-Diffusers"]
    assert "backend" not in by_name["LTX-2.5-Diffusers"]


def test_hf_cache_discovers_lightricks_ltx25_snapshot(tmp_path):
    cache = tmp_path / "hub"
    snapshot = cache / "models--Lightricks--LTX-2.5-Diffusers" / "snapshots" / "revision123"
    _model_index(snapshot, "LTX2ImageToVideoPipeline")

    items = discover_hf_video_models(cache)

    assert len(items) == 1
    item = items[0]
    assert item["name"] == "Lightricks/LTX-2.5-Diffusers"
    assert item["repo_id"] == "Lightricks/LTX-2.5-Diffusers"
    assert item["location"] == "hf_cache"
    assert item["capabilities"]["image_to_video"] is True
    assert item["capabilities"]["audio_output"] is True
    assert item["constraints"]["dimension_multiple"] == 32
    assert item["supported"] is False


def test_hf_cache_ignores_non_video_diffusers_models(tmp_path):
    cache = tmp_path / "hub"
    snapshot = cache / "models--black-forest-labs--FLUX.1-dev" / "snapshots" / "revision123"
    _model_index(snapshot, "FluxPipeline")

    assert discover_hf_video_models(cache) == []


def test_unified_discovery_includes_configured_remote_model(tmp_path):
    payload = discover_video_models(
        {"model_id_or_path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers"},
        roots=[tmp_path / "missing"],
        hf_cache=tmp_path / "hub",
    )

    assert payload["count"] == 1
    item = payload["items"][0]
    assert item["location"] == "configured"
    assert item["capabilities"]["image_to_video"] is True
    assert item["supported"] is True
