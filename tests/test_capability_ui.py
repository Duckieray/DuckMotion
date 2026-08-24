from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
APP = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")


def test_ui_branding_and_model_picker_are_architecture_neutral():
    assert "Model-driven local video generation" in INDEX
    assert "Generate Wan" not in INDEX
    assert "Wan Model" not in INDEX
    assert "local Wan models" not in INDEX
    assert 'id="discovered-models"' in INDEX
    assert 'id="selected-model-summary"' in INDEX


def test_setup_persists_only_model_neutral_configuration():
    for old_id in (
        "default-width",
        "default-height",
        "default-frames",
        "default-fps",
        "default-steps",
        "default-guidance",
    ):
        assert old_id not in INDEX

    assert "runtime_backend" not in APP
    assert "gguf_transformer_path" not in APP
    assert "GGUF" not in APP
    assert "memory_policy" not in APP
    assert "model_id_or_path" in APP
    assert "models_dir" in APP
    assert "output_dir" in APP


def test_create_view_is_driven_by_public_capabilities_and_constraints():
    assert "source_image_required" in APP
    assert "image_to_video" in APP
    assert "text_to_video" in APP
    assert "negative_prompt" in APP
    assert "audio_output" in APP
    assert "dimension_multiple" in APP
    assert "frame_count_modulo" in APP
    assert "frame_count_remainder" in APP
    assert "sampling_schedule_locked" in APP


def test_i2v_stability_control_is_capability_driven_not_model_named():
    assert 'id="i2v-stability"' in INDEX
    assert "i2v_stability_modes" in APP
    assert "i2v_stability_default" in APP
    assert "payload.i2v_stability = stability" in APP
    assert "Reference stability" in INDEX
    assert "Identity stable" in INDEX
    assert "Locked shot" in INDEX


def test_hf_cache_selection_persists_canonical_repo_identity():
    assert 'item.location === "hf_cache" && item.repo_id' in APP
    assert "modelPersistedSource(activeModel)" in APP


def test_browser_never_dispatches_on_backend_or_architecture_name():
    assert "descriptor.backend" not in APP
    assert "architecture" not in APP.lower()
    assert 'backend == "wan' not in APP.lower()
    assert 'backend == "ltx' not in APP.lower()
    assert "redgraft" not in APP.lower()
