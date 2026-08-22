from __future__ import annotations

import json
from pathlib import Path

import pytest

import storage_runtime as storage_module
from storage_runtime import VideoStorageRuntime


def _storage(tmp_path: Path) -> VideoStorageRuntime:
    return VideoStorageRuntime(
        config_file=tmp_path / "state" / "config.json",
        jobs_file=tmp_path / "state" / "jobs.json",
        staging_dir=tmp_path / "staging",
        default_output_dir=tmp_path / "videos",
    )


def test_config_is_model_neutral_and_has_no_implicit_wan(tmp_path, monkeypatch):
    monkeypatch.delenv("DUCKMOTION_MODEL_ID_OR_PATH", raising=False)
    monkeypatch.delenv("DUCKMOTION_MODELS_DIR", raising=False)
    monkeypatch.delenv("DUCKMOTION_OUTPUT_DIR", raising=False)
    storage = _storage(tmp_path)

    assert storage.load_config() == {
        "model_id_or_path": "",
        "models_dir": "",
        "output_dir": "",
    }

    storage.save_config(
        {
            "model_id_or_path": "Lightricks/LTX-2.5-Diffusers",
            "models_dir": "/models",
            "output_dir": "/outputs",
            "runtime_backend": "should-not-persist",
            "default_frames": 999,
        }
    )
    raw = json.loads(storage.config_file.read_text(encoding="utf-8"))
    assert set(raw) == {"model_id_or_path", "models_dir", "output_dir"}


def test_job_persistence_queue_snapshot_and_cancel_are_generic(tmp_path):
    storage = _storage(tmp_path)
    storage.upsert_job(
        {
            "job_id": "dm_one",
            "status": "queued",
            "created_at": 1.0,
            "cancel_requested": False,
        }
    )
    storage.upsert_job(
        {
            "job_id": "dm_two",
            "status": "running",
            "created_at": 2.0,
            "cancel_requested": False,
        }
    )

    assert [row["job_id"] for row in storage.list_jobs()] == ["dm_two", "dm_one"]
    assert storage.queue_snapshot() == {"queued": 1, "running": 1, "cancel_requested": 0}

    result = storage.cancel_job("dm_one")
    assert result["ok"] is True
    assert result["job"]["status"] == "canceled"
    assert result["job"]["cancel_requested"] is True
    assert storage.queue_snapshot()["cancel_requested"] == 1


def test_direct_input_path_validates_image_suffix(tmp_path):
    storage = _storage(tmp_path)
    image = tmp_path / "input.png"
    image.write_bytes(b"png")
    assert storage.resolve_input_image(str(image)) == image.resolve()

    bad = tmp_path / "input.txt"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported input image type"):
        storage.resolve_input_image(str(bad))


def test_staging_and_gallery_do_not_use_wan_helpers(tmp_path, monkeypatch):
    storage = _storage(tmp_path)
    monkeypatch.setattr(storage_module, "to_web_path", lambda path: f"/web/{Path(path).name}")

    item = storage.stage_bytes("source image.png", b"payload")
    assert item["name"].endswith("source_image.png")
    assert storage.list_staging()[0]["name"] == item["name"]

    output_root = storage.resolve_output_dir({})
    run = output_root / "run-1"
    run.mkdir()
    (run / "result.mp4").write_bytes(b"video")
    (run / "poster.jpg").write_bytes(b"poster")
    (run / "meta.json").write_text(
        json.dumps({"run_id": "run-1", "job_id": "dm_one", "created_at": 123.0}),
        encoding="utf-8",
    )

    gallery = storage.scan_gallery({}, 10)
    assert len(gallery) == 1
    assert gallery[0]["run_id"] == "run-1"
    assert gallery[0]["job_id"] == "dm_one"
    assert gallery[0]["video"] == "/web/result.mp4"
    assert gallery[0]["poster"] == "/web/poster.jpg"


def test_clear_jobs_removes_persisted_rows(tmp_path):
    storage = _storage(tmp_path)
    storage.upsert_job({"job_id": "dm_one", "status": "completed", "created_at": 1.0})
    storage.clear_jobs()
    assert storage.list_jobs() == []
