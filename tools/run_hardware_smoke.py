#!/usr/bin/env python3
"""Run DuckMotion's hardware smoke matrix against its live WebbDuck plugin API.

No model is downloaded by this tool. Without --execute it performs only
readiness/weight preflight. With --execute it selects models through the public
config API, submits normal jobs, polls persisted job state, and records outputs.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request
import uuid


PROMPT = "A small clockwork duck walks across a wooden workbench while the camera slowly tracks beside it"


def _json_request(url: str, *, method: str = "GET", payload: dict | None = None, timeout: float = 60.0):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {raw[-2000:]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to reach {url}: {exc}") from exc
    return json.loads(raw) if raw else {}


def _multipart(files: dict[str, Path]) -> tuple[bytes, str]:
    boundary = f"----duckmotion-smoke-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for key, path in files.items():
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"; filename="{path.name}"\r\n'.encode(),
                f"Content-Type: {mime}\r\n\r\n".encode(),
                path.read_bytes(),
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def _upload_image(api_base: str, image_path: Path) -> dict:
    body, boundary = _multipart({"image": image_path})
    request = urllib.request.Request(
        f"{api_base}/staging/upload",
        data=body,
        headers={"Accept": "application/json", "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Staging upload failed: HTTP {exc.code}: {raw[-1200:]}") from exc


def _make_source(root: Path) -> Path:
    from PIL import Image, ImageDraw

    path = root / "duckmotion_source.png"
    img = Image.new("RGB", (1024, 640), (36, 48, 62))
    draw = ImageDraw.Draw(img)
    draw.rectangle((160, 300, 864, 540), fill=(111, 76, 43))
    draw.ellipse((390, 150, 650, 410), fill=(218, 169, 58))
    draw.text((370, 560), "DuckMotion smoke", fill=(245, 245, 245))
    img.save(path)
    return path


def _find_model(items: list[dict], explicit: str | None, patterns: tuple[str, ...]) -> dict | None:
    if explicit:
        target = explicit.lower()
        for item in items:
            if str(item.get("name") or "").lower() == target or str(item.get("source") or "").lower() == target:
                return item
        return None
    for item in items:
        text = f"{item.get('name', '')} {item.get('source', '')}".lower()
        if all(pattern in text for pattern in patterns):
            return item
    return None


def _runtime_summary(item: dict | None) -> dict:
    if not item:
        return {"ready": False, "reason": "model not discovered"}
    runtime = dict(item.get("runtime") or {})
    return {
        "ready": bool(item.get("ready")),
        "reason": runtime.get("reason"),
        "python": runtime.get("python"),
        "torch": runtime.get("torch"),
        "diffusers": runtime.get("diffusers"),
        "cuda_available": runtime.get("cuda_available"),
        "gpu_name": runtime.get("gpu_name"),
        "compute_capability": runtime.get("compute_capability"),
        "vram_gb": runtime.get("vram_gb"),
        "weights": item.get("weights"),
    }


def _model_identity(item: dict) -> str:
    return str(item.get("source") or item.get("name") or "").strip()


def _select_model(api_base: str, item: dict) -> None:
    identity = _model_identity(item)
    if not identity:
        raise RuntimeError("Selected smoke model has no public source identity")
    _json_request(f"{api_base}/config", method="POST", payload={"model_id_or_path": identity}, timeout=30)


def _wait_job(api_base: str, job_id: str, timeout: float) -> tuple[dict, float]:
    started = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        payload = _json_request(f"{api_base}/engine/jobs/{job_id}", timeout=30)
        job = payload.get("job") or {}
        status = str(job.get("status") or "")
        if status == "completed":
            outputs = job.get("outputs") or []
            if not outputs:
                raise RuntimeError("DuckMotion job completed without an output artifact")
            return job, time.monotonic() - started
        if status in {"failed", "canceled", "cancelled"}:
            raise RuntimeError(str(job.get("error") or f"job ended as {status}"))
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for DuckMotion job {job_id}")


def _submit(api_base: str, payload: dict, timeout: float) -> tuple[dict, float]:
    response = _json_request(f"{api_base}/engine/generate", method="POST", payload=payload, timeout=60)
    job = response.get("job") or {}
    job_id = str(job.get("job_id") or "")
    if not job_id:
        raise RuntimeError(f"Generation submission returned no job id: {response}")
    return _wait_job(api_base, job_id, timeout)


def _rows(args, items: list[dict], image_path: str | None) -> list[dict]:
    wan5 = _find_model(items, args.wan_5b_model, ("wan2.2", "ti2v", "5b"))
    wan_i2v = _find_model(items, args.wan_i2v_model, ("wan2.2", "i2v", "a14b"))
    ltx = _find_model(items, args.ltx_model, ("ltx-2.5",))

    def row(name, model, payload, *, heavy=False, optional=False):
        return {"row": name, "model": model, "payload": payload, "heavy": heavy, "optional": optional}

    rows: list[dict] = []
    if wan5:
        rows.extend(
            [
                row("wan-ti2v-5b-canary", wan5, {"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 33, "fps": 24, "num_inference_steps": 12, "guidance_scale": 5.0, "seed": 0}),
                row("wan-ti2v-5b-default", wan5, {"prompt": PROMPT, "width": 1280, "height": 704, "num_frames": 121, "fps": 24, "num_inference_steps": 50, "guidance_scale": 5.0, "seed": 0}, heavy=True),
            ]
        )
    else:
        rows.append(row("wan-ti2v-5b", None, {}))

    if wan_i2v:
        rows.append(
            row(
                "wan-i2v-a14b-canary",
                wan_i2v,
                {"prompt": PROMPT, "image_path": image_path, "width": 512, "height": 320, "num_frames": 17, "fps": 16, "num_inference_steps": 12, "guidance_scale": 5.0, "seed": 0},
                heavy=True,
                optional=True,
            )
        )
    else:
        rows.append(row("wan-i2v-a14b-canary", None, {}, heavy=True, optional=True))

    if ltx:
        rows.extend(
            [
                row("ltx25-t2v-canary", ltx, {"prompt": PROMPT, "width": 768, "height": 512, "num_frames": 33, "fps": 24, "seed": 0}),
                row("ltx25-i2v-canary", ltx, {"prompt": PROMPT, "image_path": image_path, "width": 768, "height": 512, "num_frames": 33, "fps": 24, "seed": 0}),
                row("ltx25-t2v-default", ltx, {"prompt": PROMPT, "width": 1536, "height": 1024, "num_frames": 121, "fps": 24, "seed": 0}, heavy=True),
            ]
        )
    else:
        rows.append(row("ltx25", None, {}))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DuckMotion's hardware smoke matrix against a live WebbDuck plugin API.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8010/plugins/web/duckmotion/api")
    parser.add_argument("--execute", action="store_true", help="Actually submit video jobs. Without this flag only preflight is performed.")
    parser.add_argument("--include-heavy", action="store_true", help="Allow reference-size LTX/Wan rows and the A14B feasibility row.")
    parser.add_argument("--only", action="append", default=[], help="Run only named row(s); may be repeated.")
    parser.add_argument("--wan-5b-model", help="Override the auto-detected Wan2.2 TI2V-5B public model identity.")
    parser.add_argument("--wan-i2v-model", help="Override the auto-detected Wan2.2 I2V A14B public model identity.")
    parser.add_argument("--ltx-model", help="Override the auto-detected LTX-2.5 public model identity.")
    parser.add_argument("--timeout", type=float, default=14400.0)
    parser.add_argument("--report-dir", type=Path, default=Path("smoke_reports"))
    args = parser.parse_args()
    api_base = args.api_base.rstrip("/")

    readiness = _json_request(f"{api_base}/runtime-readiness", timeout=60)
    items = list(readiness.get("items") or [])
    if not items:
        raise SystemExit("No video models were returned by DuckMotion /runtime-readiness.")

    original_config = (_json_request(f"{api_base}/config", timeout=30).get("config") or {})
    args.report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = args.report_dir / f"duckmotion_{stamp}.json"
    report = {
        "kind": "duckmotion-hardware-smoke",
        "started_at": time.time(),
        "api_base": api_base,
        "execute": args.execute,
        "include_heavy": args.include_heavy,
        "rows": [],
    }

    staged_item: dict | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="duckmotion_smoke_") as tmp:
            source = _make_source(Path(tmp))
            image_path = None
            if args.execute:
                staged = _upload_image(api_base, source)
                staged_item = staged.get("item") or {}
                image_path = str(staged_item.get("path") or staged_item.get("web_path") or "")
                if not image_path:
                    raise RuntimeError(f"Staging returned no usable image path: {staged}")

            rows = _rows(args, items, image_path)
            if args.only:
                selected = set(args.only)
                rows = [row for row in rows if row["row"] in selected]

            for spec in rows:
                model = spec.get("model")
                entry = {
                    "row": spec["row"],
                    "model": model.get("name") if model else None,
                    "runtime": _runtime_summary(model),
                    "heavy": bool(spec.get("heavy")),
                    "status": "pending",
                }
                report["rows"].append(entry)
                if model is None:
                    entry["status"] = "skipped" if spec.get("optional") else "blocked"
                    entry["reason"] = "required smoke target is not discovered"
                    print(f"{entry['status'].upper():7} {spec['row']}: {entry['reason']}")
                    continue
                weights = model.get("weights") or {}
                if weights.get("present") is not True:
                    entry["status"] = "blocked"
                    entry["reason"] = f"weights are not fully cached locally: {weights}"
                    print(f"BLOCKED {spec['row']}: {entry['reason']}")
                    continue
                if not model.get("ready"):
                    entry["status"] = "blocked"
                    entry["reason"] = str((model.get("runtime") or {}).get("reason") or "runtime preflight is not ready")
                    print(f"BLOCKED {spec['row']}: {entry['reason']}")
                    continue
                if spec.get("heavy") and not args.include_heavy:
                    entry["status"] = "skipped"
                    entry["reason"] = "heavy/reference row requires --include-heavy"
                    print(f"SKIPPED {spec['row']}: {entry['reason']}")
                    continue
                if not args.execute:
                    entry["status"] = "ready"
                    print(f"READY   {spec['row']}: {entry['model']}")
                    continue

                try:
                    _select_model(api_base, model)
                    health = _json_request(f"{api_base}/health", timeout=60)
                    if not bool((health.get("ready") or {}).get("engine_ready")):
                        raise RuntimeError(f"selected model health gate failed: {health.get('reasons')}")
                    job, seconds = _submit(api_base, spec["payload"], args.timeout)
                    entry.update(
                        {
                            "status": "passed",
                            "elapsed_seconds": round(seconds, 3),
                            "params": job.get("params"),
                            "outputs": job.get("outputs"),
                            "runtime_profile": job.get("runtime_profile"),
                        }
                    )
                    print(f"PASSED  {spec['row']}: {seconds:.1f}s")
                except Exception as exc:
                    entry["status"] = "failed"
                    entry["error"] = str(exc)
                    print(f"FAILED  {spec['row']}: {exc}")
                finally:
                    try:
                        _json_request(f"{api_base}/engine/unload", method="POST", payload={}, timeout=120)
                    except Exception as exc:
                        entry.setdefault("warnings", []).append(f"unload failed: {exc}")
                    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        if args.execute:
            try:
                _json_request(
                    f"{api_base}/config",
                    method="POST",
                    payload={
                        "model_id_or_path": original_config.get("model_id_or_path", ""),
                        "models_dir": original_config.get("models_dir", ""),
                        "output_dir": original_config.get("output_dir", ""),
                    },
                    timeout=30,
                )
            except Exception as exc:
                report.setdefault("warnings", []).append(f"failed to restore original DuckMotion config: {exc}")
            if staged_item and staged_item.get("name"):
                try:
                    _json_request(f"{api_base}/staging/{staged_item['name']}", method="DELETE", timeout=30)
                except Exception as exc:
                    report.setdefault("warnings", []).append(f"failed to remove staged smoke image: {exc}")

    report["finished_at"] = time.time()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}")
    failed = any(row["status"] in {"failed", "blocked"} for row in report["rows"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
