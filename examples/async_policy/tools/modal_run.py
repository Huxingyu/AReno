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
import uuid
from pathlib import Path

BASELINE_SHA = "48d07c54051c41bf36218f99bce1c3697e9ba63c"


def collect_artifacts(output: Path, destination: Path, *, reports_only: bool) -> bytes:
    """Archive a small run or its reports; large checkpoints stay in the volume."""
    import tarfile

    paths = sorted(path for path in output.rglob("*") if path.is_file())
    with tarfile.open(destination, mode="w:gz", compresslevel=1) as archive:
        for path in paths:
            if not reports_only or path.suffix not in {".safetensors", ".pt", ".bin"}:
                archive.add(path, arcname=str(path.relative_to(output)))
    return destination.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "baseline", "async", "trace", "faults", "extended-faults", "full", "compiled", "graphs", "regression", "resume", "resume-full", "example", "bench-41", "bench-42", "bench-43", "benchmark", "reevaluate"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Preview resources without contacting Modal")
    parser.add_argument("--task-timeout-s", type=int, help="Bound one remote task, including artifact collection")
    parser.add_argument("--benchmark-args", nargs=argparse.REMAINDER, default=[],
                        help="Forward benchmark/reevaluation options; place last")
    args = parser.parse_args()
    if args.benchmark_args and args.phase not in {"benchmark", "reevaluate"}:
        parser.error("--benchmark-args requires --phase benchmark or reevaluate")
    if args.phase == "benchmark" and ("--seed" not in args.benchmark_args
                                      or any(flag in args.benchmark_args for flag in ("--model-path", "--output-dir"))):
        parser.error("benchmark arguments require --seed and must not override model/output paths")
    if args.phase == "reevaluate" and ("--archive" not in args.benchmark_args
                                       or any(flag in args.benchmark_args for flag in ("--model-path", "--output-dir", "--archive-root"))):
        parser.error("reevaluation requires --archive and must not override model/output/archive-root paths")
    task_timeout = args.task_timeout_s if args.task_timeout_s is not None else (
        1800 if args.phase.startswith("bench") else 1200
    )
    if not 300 <= task_timeout <= 7200:
        parser.error("--task-timeout-s must be between 300 and 7200")
    subprocess_timeout = task_timeout - 150
    gpu_request = "L4" if args.phase == "reevaluate" else "L4:2"
    root = Path(__file__).resolve().parents[3]
    if args.dry_run:
        print(json.dumps({"phase": args.phase, "gpu": None if args.phase == "prepare" else gpu_request,
                          "task_timeout_s": task_timeout, "subprocess_timeout_s": subprocess_timeout,
                          "benchmark_args": args.benchmark_args, "remote_started": False}, indent=2))
        return 0
    import modal

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
    if (args.output_dir / "request.json").exists():
        raise FileExistsError("use a fresh output directory for each remote request")
    run_id = uuid.uuid4().hex[:12]
    (args.output_dir / "request.json").write_text(json.dumps({
        "sha": sha, "baseline_sha": BASELINE_SHA, "branch": branch, "phase": args.phase,
        "gpu": None if args.phase == "prepare" else gpu_request,
        "gpu_task_timeout_s": task_timeout, "subprocess_timeout_s": subprocess_timeout,
        "model": "Qwen/Qwen3-0.6B", "model_hub": "modelscope",
        "benchmark_args": args.benchmark_args,
        "run_id": run_id,
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

    def run_remote(phase: str, source_sha: str, source_branch: str, source_url: str,
                   benchmark_args: list[str], run_id: str) -> dict:
        import json
        import os
        import signal
        import subprocess
        import sys
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
        reports_only = phase in {"full", "resume-full", "reevaluate"}
        # Retain partial evidence if Modal preempts a container. Large states
        # also avoid an expensive archive/copy after GPU work has finished.
        output = storage / f"{phase}-{source_sha[:12]}-{run_id}"
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "examples/async_policy/tools/gpu_run.py", "--mode", phase,
            "--model-path", model["model_path"], "--output-dir", str(output),
        ]
        if phase == "regression":
            baseline = workspace.parent / "areno-upstream"
            subprocess.run(["git", "worktree", "add", "--detach", str(baseline), BASELINE_SHA],
                           cwd=workspace, check=True)
            # Only the already-built, identical extension is reused. Python
            # source imports and git status are checked in each subprocess.
            extensions = list((workspace / "areno" / "accel").glob("_areno_accel*.so"))
            if len(extensions) != 1:
                raise RuntimeError("expected one upstream-built CUDA extension")
            (baseline / "areno" / "accel" / extensions[0].name).symlink_to(extensions[0])
            command = [sys.executable, "tests/async_policy_validation/sdk_regression.py",
                       "--baseline-root", str(baseline), "--model-path", model["model_path"],
                       "--data-path", str(workspace / "examples/async_policy/smoke_prompts.jsonl"),
                       "--output-dir", str(output)]
        if phase.startswith("bench-"):
            command = [sys.executable, "examples/async_policy/tools/benchmark_run.py",
                       "--seed", phase.split("-")[1], "--model-path", model["model_path"],
                       "--output-dir", str(output), "--attn-backend", "flash"]
        if phase == "benchmark":
            command = [sys.executable, "examples/async_policy/tools/benchmark_run.py",
                       "--model-path", model["model_path"], "--output-dir", str(output), *benchmark_args]
        if phase == "reevaluate":
            command = [sys.executable, "examples/async_policy/tools/reevaluate_run.py",
                       "--model-path", model["model_path"], "--output-dir", str(output),
                       "--archive-root", str(storage), *benchmark_args]
        if phase in {"resume", "resume-full"}:
            command = [sys.executable, "examples/async_policy/tools/resume_run.py",
                       "--model-path", model["model_path"], "--output-dir", str(output)]
            if phase == "resume-full":
                command.append("--full-parameters")
        if phase == "example":
            command = [sys.executable, "examples/async_policy/tools/example_run.py",
                       "--model-path", model["model_path"], "--output-dir", str(output)]
        archive_suffix = ".reports.tar.gz" if reports_only else ".tar.gz"
        archive_path = storage / f"{phase}-{source_sha}-{run_id}{archive_suffix}"
        metadata = {"command": command, "source_sha": source_sha, "phase": phase,
                    "run_id": run_id, "gpu_request": gpu_request, "volume_artifact": archive_path.name,
                    "volume_output_dir": output.name,
                    "downloaded_artifact_scope": "reports" if reports_only else "all"}
        started = time.monotonic()
        process = None
        try:
            # A preemption retry must preserve the first attempt's evidence.
            # Return the partial artifacts and require a fresh request instead
            # of overwriting logs or silently repeating paid training.
            with (output / "console.log").open("x") as log:
                process = subprocess.Popen(command, cwd=workspace, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
                try:
                    metadata["return_code"] = process.wait(timeout=subprocess_timeout)
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
        sys.path.insert(0, str(workspace))
        from examples.async_policy.tools.modal_run import collect_artifacts

        # Full-model evidence is already in the volume; only archive reports.
        artifact = collect_artifacts(output, archive_path, reports_only=reports_only)
        volume.commit()
        return {**metadata, "artifact": artifact}

    resources = dict(image=image, volumes={volume_root: volume}, max_containers=1, min_containers=0,
                     scaledown_window=2, retries=0, serialized=True, include_source=False)
    if args.phase == "prepare":
        remote = app.function(name="prepare", cpu=2, memory=8192, timeout=task_timeout, **resources)(run_remote)
    else:
        remote = app.function(name="reevaluate" if args.phase == "reevaluate" else "dual_l4_test",
                              gpu=gpu_request, cpu=4, memory=16384,
                              timeout=task_timeout, startup_timeout=300, **resources)(run_remote)
    with modal.enable_output(), app.run():
        result = remote.remote(args.phase, sha, branch, remote_url, args.benchmark_args, run_id)
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
