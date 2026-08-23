#!/usr/bin/env python3
"""Create/update DuckMotion's isolated model runtime environments.

This installs runtime libraries only. It never downloads model weights.
The LTX ConvRot runtime additionally owns a pinned Comfy core checkout used as a
Python library by the isolated worker; it never starts a ComfyUI server.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
COMFYUI_COMMIT = "9db05e0e1f035d1902ffc256fe7a336e549ced34"
COMFYUI_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
RUNTIMES = {
    "wan": ("DUCKMOTION_WAN_PYTHON", ROOT / "runtime_requirements" / "wan.txt"),
    "ltx25": ("DUCKMOTION_LTX_PYTHON", ROOT / "runtime_requirements" / "ltx25.txt"),
    "ltx25_convrot": (
        "DUCKMOTION_LTX_CONVROT_PYTHON",
        ROOT / "runtime_requirements" / "ltx25_convrot.txt",
    ),
}


def run(cmd: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(shlex.quote(str(part)) for part in cmd))
    if not dry_run:
        subprocess.check_call(cmd)


def python_in_venv(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def prepare_comfy_checkout(env_root: Path, *, dry_run: bool) -> Path:
    comfy_root = env_root / "comfyui"
    git_dir = comfy_root / ".git"
    if not git_dir.exists():
        if not dry_run:
            comfy_root.mkdir(parents=True, exist_ok=True)
        run(["git", "-C", str(comfy_root), "init"], dry_run=dry_run)
        run(["git", "-C", str(comfy_root), "remote", "add", "origin", COMFYUI_REPO], dry_run=dry_run)
    run(["git", "-C", str(comfy_root), "fetch", "--depth", "1", "origin", COMFYUI_COMMIT], dry_run=dry_run)
    run(["git", "-C", str(comfy_root), "checkout", "--detach", "--force", "FETCH_HEAD"], dry_run=dry_run)
    return comfy_root


def prepare(
    runtime: str,
    *,
    root: Path,
    base_python: str,
    torch_version: str,
    torchvision_version: str,
    torchaudio_version: str,
    torch_index: str,
    dry_run: bool,
) -> tuple[str, Path]:
    env_var, requirements = RUNTIMES[runtime]
    env_root = root / runtime
    python_path = python_in_venv(env_root)
    if not python_path.exists() or dry_run:
        env_root.parent.mkdir(parents=True, exist_ok=True)
        run([base_python, "-m", "venv", str(env_root)], dry_run=dry_run)

    runtime_python = str(python_path)
    run([runtime_python, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], dry_run=dry_run)
    torch_packages = [
        f"torch=={torch_version}",
        f"torchvision=={torchvision_version}",
    ]
    if runtime == "ltx25_convrot":
        torch_packages.append(f"torchaudio=={torchaudio_version}")
    run(
        [
            runtime_python,
            "-m",
            "pip",
            "install",
            *torch_packages,
            "--index-url",
            f"https://download.pytorch.org/whl/{torch_index}",
        ],
        dry_run=dry_run,
    )
    run([runtime_python, "-m", "pip", "install", "-r", str(requirements)], dry_run=dry_run)
    if runtime == "ltx25_convrot":
        comfy_root = prepare_comfy_checkout(env_root, dry_run=dry_run)
        print(f"Pinned Comfy core: {comfy_root} @ {COMFYUI_COMMIT}")
    return env_var, python_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", choices=["all", *RUNTIMES])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.getenv("DUCKMOTION_RUNTIME_HOME", "~/.local/share/duckmotion/runtimes")).expanduser(),
        help="Directory that owns isolated virtual environments.",
    )
    parser.add_argument("--python", default=sys.executable, help="Base Python used to create venvs.")
    parser.add_argument("--torch-version", default="2.12.1")
    parser.add_argument("--torchvision-version", default="0.27.1")
    parser.add_argument("--torchaudio-version", default="2.12.1")
    parser.add_argument("--torch-index", default="cu130", help="PyTorch wheel channel, e.g. cu130 or cu132.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = list(RUNTIMES) if args.runtime == "all" else [args.runtime]
    exports: list[tuple[str, Path]] = []
    for runtime in selected:
        print(f"\n== {runtime} ==")
        exports.append(
            prepare(
                runtime,
                root=args.root.expanduser(),
                base_python=args.python,
                torch_version=args.torch_version,
                torchvision_version=args.torchvision_version,
                torchaudio_version=args.torchaudio_version,
                torch_index=args.torch_index,
                dry_run=args.dry_run,
            )
        )

    print("\nAdd these to the WebbDuck/DuckMotion launch environment:")
    for env_var, python_path in exports:
        print(f"export {env_var}={shlex.quote(str(python_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
