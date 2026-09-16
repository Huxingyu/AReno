"""Bounded, committed-source Modal runner for the dual-L4 acceptance probes.

Install Modal in a separate tools environment, then run this file with
``--phase prepare`` before ``--phase baseline`` or ``--phase async``.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
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
    parser.add_argument("--phase", choices=("prepare", "prepare-kernels", "kernel-check", "checkpoint-inventory", "baseline", "async", "trace", "faults", "extended-faults", "full", "compiled", "graphs", "regression", "resume", "resume-full", "example", "bench-41", "bench-42", "bench-43", "benchmark", "reevaluate", "checkpoint-eval"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Preview resources without contacting Modal")
    parser.add_argument("--task-timeout-s", type=int, help="Bound one remote task, including artifact collection")
    parser.add_argument("--controller-timeout-s", type=int,
                        help="Bound provisioning, execution and automatic retries together")
    parser.add_argument("--budget-file", type=Path, help="Shared conservative spending ledger; required to execute")
    parser.add_argument("--budget-usd", type=float, help="Initialize the shared ledger; existing limits cannot change")
    parser.add_argument("--run-id", help="Stable remote attempt ID; replay never silently retrains an existing attempt")
    parser.add_argument("--benchmark-args", nargs=argparse.REMAINDER, default=[],
                        help="Forward benchmark/reevaluation options; place last")
    args = parser.parse_args()
    if args.benchmark_args and args.phase not in {"benchmark", "reevaluate", "checkpoint-eval", "resume-full", "checkpoint-inventory", "kernel-check"}:
        parser.error("forwarded arguments require benchmark, reevaluate, checkpoint-eval or resume-full")
    if args.phase == "benchmark" and ("--seed" not in args.benchmark_args
                                      or any(flag in args.benchmark_args for flag in ("--model-path", "--output-dir"))):
        parser.error("benchmark arguments require --seed and must not override model/output paths")
    if args.phase == "reevaluate" and ("--archive" not in args.benchmark_args
                                       or any(flag in args.benchmark_args for flag in ("--model-path", "--output-dir", "--archive-root"))):
        parser.error("reevaluation requires --archive and must not override model/output/archive-root paths")
    if args.phase == "checkpoint-eval":
        if "--training-output" not in args.benchmark_args:
            parser.error("checkpoint evaluation requires --training-output (a relative Volume directory)")
        if any(flag in args.benchmark_args for flag in ("--model-path", "--output-dir", "--cache-dir")):
            parser.error("checkpoint evaluation must not override model/output/cache paths")
    if args.phase == "resume-full" and any(
        word.split("=", 1)[0] in {"--model-path", "--output-dir"} for word in args.benchmark_args
    ):
        parser.error("resume arguments must not override model/output paths")
    task_timeout = args.task_timeout_s if args.task_timeout_s is not None else (
        1800 if args.phase.startswith("bench") else 1200
    )
    if not 300 <= task_timeout <= 7200:
        parser.error("--task-timeout-s must be between 300 and 7200")
    subprocess_timeout = task_timeout - 150
    controller_timeout = args.controller_timeout_s or task_timeout + 300
    if not 30 <= controller_timeout <= 7500:
        parser.error("controller timeout must be between 30 and 7500 seconds")
    cpu_only = args.phase in {"prepare", "prepare-kernels", "checkpoint-inventory"}
    gpu_request = "L4" if args.phase in {"reevaluate", "checkpoint-eval", "kernel-check"} else "L4:2"
    root = Path(__file__).resolve().parents[3]
    if args.dry_run:
        print(json.dumps({"phase": args.phase, "gpu": None if cpu_only else gpu_request,
                          "task_timeout_s": task_timeout, "subprocess_timeout_s": subprocess_timeout,
                          "controller_timeout_s": controller_timeout,
                          "benchmark_args": args.benchmark_args, "remote_started": False}, indent=2))
        return 0
    if args.budget_file is None:
        parser.error("--budget-file is required before starting paid Modal work")
    sys.path.insert(0, str(root))
    from examples.async_policy.tools.campaign_state import BudgetLedger, write_json
    from examples.async_policy.tools.modal_control import invoke

    budget = BudgetLedger(args.budget_file.resolve(), args.budget_usd)
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
    run_id = args.run_id or uuid.uuid4().hex[:12]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        parser.error("run ID must be a short, safe directory component")
    write_json(args.output_dir / "request.json", {
        "sha": sha, "baseline_sha": BASELINE_SHA, "branch": branch, "phase": args.phase,
        "gpu": None if cpu_only else gpu_request,
        "gpu_task_timeout_s": task_timeout, "subprocess_timeout_s": subprocess_timeout,
        "controller_timeout_s": controller_timeout,
        "model": "Qwen/Qwen3-0.6B", "model_hub": "modelscope",
        "benchmark_args": args.benchmark_args,
        "run_id": run_id,
    })

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
    app_name = f"areno-async-policy-{run_id}"
    app = modal.App(app_name)

    def run_remote(phase: str, source_sha: str, source_branch: str, source_url: str,
                   benchmark_args: list[str], run_id: str, deadline_epoch: float) -> dict:
        import hashlib
        import json
        import os
        import shutil
        import signal
        import subprocess
        import sys
        import threading
        import time
        import traceback
        from pathlib import Path

        workspace = Path.home() / "areno"
        storage = Path(volume_root)
        if time.time() >= deadline_epoch - 15:
            return {"return_code": 124, "phase": phase, "deadline_expired": True}
        output = storage / f"{phase}-{source_sha[:12]}-{run_id}"
        output.mkdir(parents=True, exist_ok=True)
        marker = output / "attempt-start.json"
        try:
            with marker.open("x") as stream:
                json.dump({"source_sha": source_sha, "run_id": run_id,
                           "benchmark_args": benchmark_args,
                           "started_at": time.time(), "deadline_epoch": deadline_epoch}, stream)
            # No model work starts until the retry guard is durable remotely.
            volume.commit()
        except FileExistsError:
            original = json.loads(marker.read_text())
            if original["source_sha"] != source_sha or original["benchmark_args"] != benchmark_args:
                raise ValueError("existing remote attempt has different settings")
            previous = output / "modal-task.json"
            metadata = json.loads(previous.read_text()) if previous.exists() else {"return_code": 75}
            artifact_name = metadata.get("volume_artifact")
            archive = storage / artifact_name if artifact_name else None
            if archive and archive.is_file() and metadata.get("artifact_sha256"):
                artifact = archive.read_bytes()
                if hashlib.sha256(artifact).hexdigest() != metadata["artifact_sha256"]:
                    raise ValueError("stored remote report archive changed")
                return {**metadata, "artifact": artifact, "replayed": True}
            return {**metadata, "run_id": run_id, "phase": phase, "replayed": True,
                    "return_code": 75,
                    "volume_output_dir": output.name,
                    "replay_reason": "attempt already started; inspect evidence or use an explicit new attempt"}
        subprocess.run(["git", "fetch", "--depth=1", source_url, source_branch], cwd=workspace, check=True)
        fetched = subprocess.check_output(["git", "rev-parse", "FETCH_HEAD"], cwd=workspace, text=True).strip()
        if fetched != source_sha:
            raise RuntimeError("Remote branch moved; refusing a mismatched source SHA")
        # Kernel changes are compiled in a bounded CPU job, then loaded only
        # from an exact source/runtime/hash match on the GPU host.
        changed_kernels = subprocess.check_output(
            ["git", "diff", BASELINE_SHA, source_sha, "--", "areno/accel", "setup.py"], cwd=workspace, text=True,
        )
        subprocess.run(["git", "checkout", "--detach", source_sha], cwd=workspace, check=True)
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=workspace, text=True).strip():
            raise RuntimeError("Remote source checkout is dirty")
        sys.path.insert(0, str(workspace))
        from examples.async_policy.tools.campaign_state import file_manifest, sha256_file, write_json

        if phase == "checkpoint-inventory":
            if len(benchmark_args) != 2 or benchmark_args[0] != "--directory":
                raise ValueError("checkpoint inventory requires --directory and a relative Volume directory")
            reference = Path(benchmark_args[1])
            if reference.is_absolute() or ".." in reference.parts:
                raise ValueError("inventory directory must remain within the Volume")
            inventory = file_manifest(storage / reference)
            write_json(output / "inventory.json", inventory)
            volume.commit()
            return {"return_code": 0, "phase": phase, "source_sha": source_sha,
                    "directory": str(reference), "inventory": inventory}

        if changed_kernels:
            import torch

            tree = subprocess.check_output(["git", "ls-tree", "-r", source_sha, "--", "areno/accel", "setup.py"], cwd=workspace)
            kernel_key = hashlib.sha256(tree).hexdigest()
            cache = storage / "compiled-kernels" / kernel_key
            metadata_path = cache / "build.json"
            cache_hit = metadata_path.is_file()
            if phase == "prepare-kernels" and not cache_hit:
                cache.mkdir(parents=True, exist_ok=True)
                try:
                    with (output / "kernel-build.log").open("x") as log:
                        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "--no-build-isolation"],
                                       cwd=workspace, stdout=log, stderr=subprocess.STDOUT, check=True,
                                       timeout=max(1, deadline_epoch - time.time() - 30))
                finally:
                    volume.commit()
                extensions = list((workspace / "areno/accel").glob("_areno_accel*.so"))
                if len(extensions) != 1:
                    raise RuntimeError("expected one freshly built AReno extension")
                binary = cache / extensions[0].name
                shutil.copy2(extensions[0], binary)
                metadata = {"source_sha": source_sha, "kernel_sha256": kernel_key,
                            "torch": str(torch.__version__), "cuda": torch.version.cuda,
                            "architecture": os.environ["TORCH_CUDA_ARCH_LIST"],
                            "binary": binary.name, "binary_sha256": sha256_file(binary)}
                write_json(metadata_path, metadata)
                volume.commit()
            if phase == "regression":
                raise RuntimeError("kernel regression needs independent baseline/candidate extensions")
            if not metadata_path.is_file():
                raise RuntimeError("changed CUDA source requires --phase prepare-kernels first")
            metadata = json.loads(metadata_path.read_text())
            binary = cache / metadata["binary"]
            if (metadata["kernel_sha256"] != kernel_key or metadata["torch"] != torch.__version__
                    or metadata["cuda"] != torch.version.cuda
                    or metadata["architecture"] != os.environ["TORCH_CUDA_ARCH_LIST"]
                    or sha256_file(binary) != metadata["binary_sha256"]):
                raise RuntimeError("cached CUDA extension does not match this source and runtime")
            if phase == "prepare-kernels":
                # Read JSON primitives so the controller needs no Torch import
                # when Modal deserializes torch.torch_version.TorchVersion.
                return {"return_code": 0, "phase": phase, "kernel_build": metadata, "cache_hit": cache_hit}
            shutil.copy2(binary, workspace / "areno/accel" / binary.name)
        elif phase == "prepare-kernels":
            return {"return_code": 0, "phase": phase, "kernel_build": "unchanged baseline image"}

        if phase == "prepare":
            from modelscope import snapshot_download

            model_path = snapshot_download("Qwen/Qwen3-0.6B", cache_dir=str(storage / "models"))
            config = json.loads((Path(model_path) / "config.json").read_text())
            metadata = {"model_path": model_path, "model_hub": "modelscope", "config": config}
            (storage / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
            volume.commit()
            return {"return_code": 0, "phase": phase, "source_sha": source_sha, **metadata}

        model = json.loads((storage / "model.json").read_text())
        reports_only = True
        # Retain partial evidence if Modal preempts a container. Large states
        # also avoid an expensive archive/copy after GPU work has finished.
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
        if phase == "checkpoint-eval":
            forwarded = list(benchmark_args)
            index = forwarded.index("--training-output") + 1
            reference = Path(forwarded[index])
            if reference.is_absolute() or ".." in reference.parts:
                raise ValueError("training output must be relative to the Volume root")
            forwarded[index] = str(storage / reference)
            command = [sys.executable, "examples/async_policy/tools/checkpoint_eval.py",
                       "--model-path", model["model_path"], "--output-dir", str(output),
                       "--cache-dir", str(storage / "evaluation-cache"), *forwarded]
        if phase in {"resume", "resume-full"}:
            command = [sys.executable, "examples/async_policy/tools/resume_run.py",
                       "--model-path", model["model_path"], "--output-dir", str(output)]
            if phase == "resume-full":
                command.append("--full-parameters")
                command.extend(benchmark_args)
        if phase == "example":
            command = [sys.executable, "examples/async_policy/tools/example_run.py",
                       "--model-path", model["model_path"], "--output-dir", str(output)]
        if phase == "kernel-check":
            command = [sys.executable, "tests/async_policy_validation/deterministic_reductions.py",
                       "--output-dir", str(output), *benchmark_args]
        archive_suffix = ".reports.tar.gz" if reports_only else ".tar.gz"
        archive_path = storage / f"{phase}-{source_sha}-{run_id}{archive_suffix}"
        metadata = {"command": command, "source_sha": source_sha, "phase": phase,
                    "run_id": run_id, "gpu_request": gpu_request, "volume_artifact": archive_path.name,
                    "volume_output_dir": output.name,
                    "downloaded_artifact_scope": "reports" if reports_only else "all"}
        started = time.monotonic()
        process = None
        persist_stop = threading.Event()

        def persist_progress():
            while not persist_stop.wait(10):
                try:
                    volume.commit()
                except Exception as exc:
                    print(f"progress commit failed: {exc}", flush=True)

        persistence = threading.Thread(target=persist_progress, daemon=True)
        persistence.start()
        try:
            # A preemption retry must preserve the first attempt's evidence.
            # Return the partial artifacts and require a fresh request instead
            # of overwriting logs or silently repeating paid training.
            with (output / "console.log").open("x") as log:
                process = subprocess.Popen(command, cwd=workspace, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
                try:
                    remaining = max(0.01, min(subprocess_timeout, deadline_epoch - time.time() - 20))
                    metadata["return_code"] = process.wait(timeout=remaining)
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
            persist_stop.set()
            persistence.join(timeout=15)
            metadata["wall_s"] = time.monotonic() - started
            (output / "modal-task.json").write_text(json.dumps(metadata, indent=2) + "\n")
        sys.path.insert(0, str(workspace))
        from examples.async_policy.tools.modal_run import collect_artifacts

        # Full-model evidence is already in the volume; only archive reports.
        artifact = collect_artifacts(output, archive_path, reports_only=reports_only)
        metadata["artifact_sha256"] = hashlib.sha256(artifact).hexdigest()
        (output / "modal-task.json").write_text(json.dumps(metadata, indent=2) + "\n")
        volume.commit()
        return {**metadata, "artifact": artifact}

    resources = dict(image=image, volumes={volume_root: volume}, max_containers=1, min_containers=0,
                     scaledown_window=2, retries=0, serialized=True, include_source=False)
    if cpu_only:
        remote = app.function(name="prepare", cpu=(4, 4), memory=(16384, 16384), timeout=task_timeout, **resources)(run_remote)
    else:
        remote = app.function(name="evaluate" if args.phase in {"reevaluate", "checkpoint-eval"} else "dual_l4_test",
                              gpu=gpu_request, cpu=(4, 4), memory=(32768, 32768),
                              timeout=task_timeout, startup_timeout=300, **resources)(run_remote)
    with modal.enable_output():
        result = invoke(app, remote, (args.phase, sha, branch, remote_url, args.benchmark_args, run_id),
                        output=args.output_dir, budget=budget, timeout_s=controller_timeout,
                        gpus=0 if cpu_only else 1 if gpu_request == "L4" else 2,
                        app_name=app_name)
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
