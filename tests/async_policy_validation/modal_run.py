"""Bounded, committed-source Modal runner for the dual-L4 acceptance probes.

Install Modal in a separate tools environment, then run this file with
``--phase prepare`` before ``--phase baseline`` or ``--phase async``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

BASELINE_SHA = "48d07c54051c41bf36218f99bce1c3697e9ba63c"


def main() -> int:
    import modal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "baseline", "async", "trace", "faults"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]

    def git(*command: str) -> str:
        return subprocess.check_output(["git", *command], cwd=root, text=True).strip()

    if git("status", "--porcelain"):
        raise RuntimeError("Commit local changes before running the remote checkout")
    sha = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    remote_url = git("remote", "get-url", "origin")
    remote_head = git("ls-remote", "origin", f"refs/heads/{branch}").split()
    if not remote_head or remote_head[0] != sha:
        raise RuntimeError("Push the committed branch before running Modal")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "request.json").write_text(json.dumps({
        "sha": sha, "baseline_sha": BASELINE_SHA, "branch": branch, "phase": args.phase,
        "gpu": "L4:2", "gpu_task_timeout_s": 1200, "subprocess_timeout_s": 1050,
        "model": "Qwen/Qwen3-0.6B", "model_hub": "modelscope",
    }, indent=2) + "\n")

    # Compile on CPU and retain the extension built from the clean upstream.
    # Pure Python candidate changes are fetched below without reinstalling.
    image = (
        modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
        .apt_install("git", "build-essential", "ninja-build", "libgomp1")
        .env({"TORCH_CUDA_ARCH_LIST": "8.9", "MAX_JOBS": "2", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
        .pip_install("psutil", "packaging", "ninja", "wheel", "setuptools")
        .run_commands(
            'git init "$HOME/areno"',
            f'git -C "$HOME/areno" fetch --depth=1 {shlex.quote(remote_url)} {BASELINE_SHA}',
            'git -C "$HOME/areno" checkout --detach FETCH_HEAD',
            'pip install -e "$HOME/areno" --no-build-isolation',
        )
        .pip_install("flash-attn==2.8.3", extra_options="--no-build-isolation")
    )
    volume_root = str(Path(tempfile.gettempdir()) / "areno-r4")
    volume = modal.Volume.from_name("areno-async-policy-r4", create_if_missing=True)
    app = modal.App("areno-async-policy-r4")

    def run_remote(phase: str, source_sha: str, source_branch: str, source_url: str) -> dict:
        import io
        import json
        import os
        import signal
        import subprocess
        import sys
        import tarfile
        import time
        import traceback
        from pathlib import Path

        workspace = Path.home() / "areno"
        storage = Path(volume_root)
        subprocess.run(["git", "fetch", "--depth=1", source_url, source_branch], cwd=workspace, check=True)
        fetched = subprocess.check_output(["git", "rev-parse", "FETCH_HEAD"], cwd=workspace, text=True).strip()
        if fetched != source_sha:
            raise RuntimeError("Remote branch moved; refusing a mismatched source SHA")
        # Reject a candidate whose CUDA source differs from the built image.
        changed_kernels = subprocess.check_output(
            ["git", "diff", BASELINE_SHA, source_sha, "--", "areno/accel", "setup.py"], cwd=workspace, text=True,
        )
        if changed_kernels:
            raise RuntimeError("CUDA build inputs changed; rebuild the image before running")
        subprocess.run(["git", "checkout", "--detach", source_sha], cwd=workspace, check=True)
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=workspace, text=True).strip():
            raise RuntimeError("Remote source checkout is dirty")

        if phase == "prepare":
            from modelscope import snapshot_download

            model_path = snapshot_download("Qwen/Qwen3-0.6B", cache_dir=str(storage / "models"))
            config = json.loads((Path(model_path) / "config.json").read_text())
            metadata = {"model_path": model_path, "model_hub": "modelscope", "config": config}
            (storage / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
            volume.commit()
            return {"return_code": 0, "phase": phase, "source_sha": source_sha, **metadata}

        model = json.loads((storage / "model.json").read_text())
        output = workspace / "runs" / "async-policy-rewrite" / "gpu" / f"{phase}-{source_sha[:12]}"
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "tests/async_policy_validation/gpu_run.py", "--mode", phase,
            "--model-path", model["model_path"], "--output-dir", str(output),
        ]
        metadata = {"command": command, "source_sha": source_sha, "phase": phase, "gpu_request": "L4:2"}
        started = time.monotonic()
        process = None
        try:
            with (output / "console.log").open("w") as log:
                process = subprocess.Popen(command, cwd=workspace, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
                try:
                    metadata["return_code"] = process.wait(timeout=1050)
                except subprocess.TimeoutExpired:
                    metadata["timeout"] = True
                    os.killpg(process.pid, signal.SIGKILL)
                    metadata["return_code"] = process.wait(timeout=10)
        except BaseException:
            metadata["runner_error"] = traceback.format_exc()
            metadata["return_code"] = 1
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        finally:
            metadata["wall_s"] = time.monotonic() - started
            (output / "modal-task.json").write_text(json.dumps(metadata, indent=2) + "\n")
        packed = io.BytesIO()
        with tarfile.open(fileobj=packed, mode="w:gz") as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=str(path.relative_to(output)))
        # Save a second copy before returning, including failed run evidence.
        artifact = packed.getvalue()
        (storage / f"{phase}-{source_sha}.tar.gz").write_bytes(artifact)
        volume.commit()
        return {**metadata, "artifact": artifact}

    resources = dict(image=image, volumes={volume_root: volume}, max_containers=1, min_containers=0,
                     scaledown_window=2, retries=0, serialized=True, include_source=False)
    if args.phase == "prepare":
        remote = app.function(name="prepare", cpu=2, memory=8192, timeout=1200, **resources)(run_remote)
    else:
        remote = app.function(name="dual_l4_test", gpu="L4:2", cpu=4, memory=16384,
                              timeout=1200, startup_timeout=300, **resources)(run_remote)
    with modal.enable_output(), app.run():
        result = remote.remote(args.phase, sha, branch, remote_url)
    artifact = result.pop("artifact", None)
    if artifact is not None:
        import io
        import tarfile

        (args.output_dir / "artifacts.tar.gz").write_bytes(artifact)
        with tarfile.open(fileobj=io.BytesIO(artifact), mode="r:gz") as archive:
            archive.extractall(args.output_dir / "artifacts", filter="data")
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["return_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
