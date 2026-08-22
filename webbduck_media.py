"""Small WebbDuck media bridge used by DuckMotion staging UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from server.storage import BASE, to_web_path

from storage_runtime import DEFAULT_OUTPUT_DIR, STAGING_DIR, SUPPORTED_IMAGE_SUFFIXES


def recent_webbduck_images(limit: int = 24) -> list[dict[str, Any]]:
    try:
        runs = [
            path
            for path in BASE.iterdir()
            if path.is_dir()
            and path.resolve() not in {STAGING_DIR.resolve(), DEFAULT_OUTPUT_DIR.resolve()}
        ]
    except Exception:
        return []
    runs.sort(key=lambda path: path.name, reverse=True)

    out: list[dict[str, Any]] = []
    for run in runs:
        try:
            images = sorted(run.iterdir(), key=lambda path: path.name)
        except OSError:
            continue
        for image in images:
            if not image.is_file() or image.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            if image.name.endswith("_upscaled.png") or ".thumb" in image.name:
                continue
            try:
                stat = image.stat()
                web_path = str(to_web_path(image))
            except Exception:
                continue
            out.append(
                {
                    "run": run.name,
                    "name": image.name,
                    "web_path": web_path,
                    "path": str(image),
                    "size_bytes": int(stat.st_size),
                    "mtime": float(stat.st_mtime),
                }
            )
            if len(out) >= max(1, int(limit)):
                return out
    return out
