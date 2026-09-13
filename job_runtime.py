"""Generic DuckMotion job orchestration shared by all video backends."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from model_runtime import VideoModelDescriptor, backend_resolver


class VideoJobCoordinator:
    """Own queue/persistence/GPU coordination without architecture branching."""

    def __init__(self, services: Any) -> None:
        self.services = services

    def submit(
        self,
        descriptor: VideoModelDescriptor,
        request: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = str(request.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt is required")

        raw_image = str(request.get("image_path") or "").strip()
        if descriptor.capabilities.source_image_required and not raw_image:
            raise ValueError("The selected model requires a source image")
        if raw_image and not descriptor.capabilities.image_to_video:
            raise ValueError("The selected model does not support image conditioning")

        source_path = self.services.resolve_input_image(raw_image) if raw_image else None
        params = self._normalize_params(descriptor, request)
        now = self.services.now()
        job_id = f"dm_{uuid.uuid4().hex[:12]}"
        row = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "cancel_requested": False,
            "progress": {"stage": "queued", "percent": 0},
            "model": descriptor.to_public_dict(),
            "config_snapshot": {
                "model_id_or_path": descriptor.source,
                "output_dir": str(self.services.resolve_output_dir(config)),
            },
            "input": {
                "image_path": str(source_path) if source_path is not None else "",
                "image_name": source_path.name if source_path is not None else "",
                "web_path": self.services.safe_web_path(source_path) if source_path is not None else None,
            },
            "params": params,
            "error": None,
            "warnings": [],
            "outputs": [],
        }
        self.services.upsert_job(row)
        threading.Thread(
            target=self._run,
            args=(job_id, descriptor, dict(config)),
            name=f"duckmotion-job-{job_id}",
            daemon=True,
        ).start()
        return row

    def _run(self, job_id: str, descriptor: VideoModelDescriptor, config: dict[str, Any]) -> None:
        row = self.services.get_job(job_id)
        if row is None or row.get("status") == "canceled":
            return

        lease_token = None
        backend = None
        try:
            row["status"] = "running"
            row["started_at"] = self.services.now()
            row["updated_at"] = self.services.now()
            row["progress"] = {"stage": "waiting_for_gpu", "percent": 5}
            self.services.upsert_job(row)

            lease_attempt = self.services.acquire_gpu_lease(
                owner="duckmotion",
                owner_kind="plugin",
                label="video_generation",
                job_id=job_id,
            )
            lease = lease_attempt.get("lease") if isinstance(lease_attempt, dict) else None
            if isinstance(lease, dict):
                lease_token = str(lease.get("token") or "").strip() or None

            # A model switch must not inherit accelerator state from a previous
            # video runtime. The resolver owns this operation; the coordinator
            # does not know which backend family held those resources.
            backend_resolver.unload_all()

            row = self.services.get_job(job_id) or row
            if row.get("cancel_requested"):
                raise RuntimeError("Generation cancelled")

            row["progress"] = {"stage": "loading_pipeline", "percent": 15}
            row["updated_at"] = self.services.now()
            self.services.upsert_job(row)

            runtime_profile = self.services.runtime_profile()
            row["runtime_profile"] = runtime_profile
            self.services.upsert_job(row)

            backend = backend_resolver.resolve(descriptor)
            run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{job_id}"
            run_dir = self.services.resolve_output_dir(config) / run_id
            request = dict(row.get("params") or {})
            request["image_path"] = str((row.get("input") or {}).get("image_path") or "")

            def is_cancelled() -> bool:
                current = self.services.get_job(job_id)
                return bool(current and current.get("cancel_requested"))

            result = backend.generate(
                descriptor,
                request,
                output_dir=run_dir,
                is_cancelled=is_cancelled,
                job=row,
                config=config,
                runtime_profile=runtime_profile,
                lease_token=lease_token,
            )
            output_info = self._normalize_output(
                descriptor,
                row,
                run_id,
                run_dir,
                result,
            )

            row = self.services.get_job(job_id) or row
            row["status"] = "completed"
            row["error"] = None
            row["outputs"] = [output_info]
            row["updated_at"] = self.services.now()
            row["finished_at"] = self.services.now()
            row["progress"] = {"stage": "completed", "percent": 100}
            self.services.upsert_job(row)
        except Exception as exc:
            row = self.services.get_job(job_id) or row
            canceled = bool(row.get("cancel_requested")) or "cancel" in str(exc).lower()
            row["status"] = "canceled" if canceled else "failed"
            row["error"] = None if canceled else str(exc or exc.__class__.__name__)
            row["updated_at"] = self.services.now()
            row["finished_at"] = self.services.now()
            row["progress"] = {
                "stage": row["status"],
                "percent": row.get("progress", {}).get("percent", 0),
            }
            self.services.upsert_job(row)
        finally:
            if backend is not None:
                try:
                    backend.unload()
                except Exception:
                    pass
            if lease_token:
                try:
                    self.services.release_gpu_lease(token=lease_token)
                except Exception:
                    pass

    def _normalize_output(
        self,
        descriptor: VideoModelDescriptor,
        row: dict[str, Any],
        run_id: str,
        run_dir: Path,
        result: Any,
    ) -> dict[str, Any]:
        if isinstance(result, dict) and isinstance(result.get("output_info"), dict):
            return result["output_info"]

        if not isinstance(result, dict):
            raise RuntimeError("Video backend returned an invalid result")

        meta_path = run_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        except Exception:
            meta = {}
        meta.update(
            {
                "plugin": "duckmotion",
                "job_id": row.get("job_id"),
                "run_id": run_id,
                "created_at": self.services.now(),
                "input": row.get("input") or {},
                "params": row.get("params") or {},
                "model": descriptor.to_public_dict(),
                "frame_count": int(result.get("frame_count") or (row.get("params") or {}).get("num_frames") or 0),
            }
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        output_info = self.services.gallery_item_from_run(run_dir)
        if output_info is None:
            raise RuntimeError("Video generation completed but no gallery artifact could be built")
        return output_info

    @staticmethod
    def _snap(value: int, multiple: int) -> int:
        return max(multiple, (int(value) // multiple) * multiple)

    def _normalize_params(self, descriptor: VideoModelDescriptor, request: dict[str, Any]) -> dict[str, Any]:
        defaults = descriptor.defaults or {}
        constraints = descriptor.constraints or {}
        multiple = max(1, int(constraints.get("dimension_multiple") or 1))
        width = self._snap(int(request.get("width") or defaults.get("width") or 832), multiple)
        height = self._snap(int(request.get("height") or defaults.get("height") or 480), multiple)

        frames = int(request.get("num_frames") or defaults.get("num_frames") or 81)
        modulo = constraints.get("frame_count_modulo")
        remainder = constraints.get("frame_count_remainder")
        if modulo is not None and remainder is not None:
            mod = max(1, int(modulo))
            rem = int(remainder)
            frames = max(rem + mod, ((max(frames, rem) - rem) // mod) * mod + rem)

        params = {
            "prompt": str(request.get("prompt") or "").strip(),
            "width": width,
            "height": height,
            "num_frames": frames,
            "fps": int(request.get("fps") or defaults.get("fps") or 16),
            "num_inference_steps": int(request.get("num_inference_steps") or defaults.get("num_inference_steps") or 30),
            "guidance_scale": float(
                request.get("guidance_scale")
                if request.get("guidance_scale") is not None
                else defaults.get("guidance_scale", 5.0)
            ),
            "seed": int(request["seed"]) if request.get("seed") is not None else None,
        }
        if descriptor.capabilities.negative_prompt:
            params["negative_prompt"] = str(request.get("negative_prompt") or "")
        return params
