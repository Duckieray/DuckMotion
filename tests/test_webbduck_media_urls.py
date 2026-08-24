from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "webbduck_media.py").read_text(encoding="utf-8")


def test_recent_webbduck_thumbnails_are_host_root_relative():
    assert 'return "/" + str(to_web_path(path)).lstrip("/")' in SOURCE
    assert "web_path = _root_web_path(image)" in SOURCE
