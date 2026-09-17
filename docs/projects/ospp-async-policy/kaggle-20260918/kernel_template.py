"""AReno async-policy campaign runner for Kaggle (2x T4, fp16).

Generated from kernel_template.py by gen.py; LANE is substituted per kernel.
Roles:
  build  - clone repo, build CUDA extension wheel for sm_75, fetch Qwen3-0.6B,
           write an fp16 copy of the model config, run one smoke job.
  laneN  - mount the build output, install wheel, run matrix.py --backend local.
  lane0  - strict full-parameter resume seed43 (deterministic branch) and the
           regression / faults / extended-faults GPU suites.
"""
import json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

LANE = "__LANE__"
BATCHED_SHA = "bb8a0637bbf13c46454fff2ed4620e330e3d95f1"
DETERMINISTIC_SHA = "17a9cb19bfaf5cc5b019afc9749f168421dcd3cc"
BASELINE_SHA = "48d07c54051c41bf36218f99bce1c3697e9ba63c"
REPO = "https://github.com/Huxingyu/AReno.git"
UPSTREAM = "https://github.com/inclusionAI/AReno.git"
MODEL_ID = "Qwen/Qwen3-0.6B"
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
BUILD_INPUT = next(iter(INPUT.glob("areno-kaggle-build*")), None)
STATE_INPUT = next(iter(INPUT.glob("areno-campaign-state*")), None)
SRC = Path("/kaggle/tmp/areno"); SRC.parent.mkdir(parents=True, exist_ok=True)
STARTED = time.time()
BUDGET_S = int(os.environ.get("ARENO_KAGGLE_BUDGET_S", str(int(10.5 * 3600))))  # stay under the 12 h session cap
LOG = WORK / f"{LANE}.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')} +{int(time.time() - STARTED)}s] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sh(cmd, cwd=None, env=None, check=True, timeout=None):
    log("$ " + (cmd if isinstance(cmd, str) else " ".join(map(str, cmd))))
    return subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env={**os.environ, **(env or {})},
                          check=check, timeout=timeout)


def remaining():
    return BUDGET_S - (time.time() - STARTED)


def gpu_info():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    log(f"GPUs: {out}")
    if len(out) < 2:
        raise SystemExit("this campaign needs two GPUs (train on cuda:0, rollout on cuda:1); pick the T4 x2 machine")
    return out


def checkout(sha, dest):
    if not (dest / ".git").exists():
        sh(["git", "init", "-q", str(dest)])
        sh(["git", "-C", str(dest), "fetch", "-q", "--depth=1", REPO, sha])
        sh(["git", "-C", str(dest), "checkout", "-q", "--detach", "FETCH_HEAD"])
    got = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    assert got == sha, (got, sha)
    return dest


def pip(*args):
    sh([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *args])


def install_runtime_deps():
    # Same pins the Modal image resolved from pyproject; flash-attn is skipped
    # (attention backend "native" never imports it) because T4 is sm_75.
    pip("transformers>=5.15,<6", "modelscope>=1.20", "safetensors>=0.4", "datasets==4.0.0", "math-verify",
        "addict>=2.4", "psutil", "packaging", "ninja", "flash-linear-attention==0.5.2", "tensorboard",
        "pydantic>=2", "fastapi>=0.110", "click>=8.1", "uvicorn>=0.27", "prompt-toolkit>=3.0", "rich>=13",
        "huggingface-hub>=0.25", "openai", "tqdm>=4.66", "av>=12", "librosa>=0.11", "soundfile>=0.13")


def torch_tag():
    import torch
    return f"torch{torch.__version__}-py{sys.version_info.major}{sys.version_info.minor}"


def fp16_model(src_model: Path, dest: Path):
    """Copy the model directory and switch torch_dtype to float16 for T4."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src_model, dest, symlinks=False)
    cfg_path = dest / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["torch_dtype"] = "float16"
    cfg.pop("dtype", None)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    (dest / "AREN0_FP16_NOTE.txt").write_text(
        "Copied from %s; torch_dtype forced to float16 because T4 (sm_75) lacks bf16.\n" % src_model)
    return dest


# ----------------------------------------------------------------------------- build
def role_build():
    gpu_info()
    install_runtime_deps()
    repo = checkout(BATCHED_SHA, SRC)
    # Confirm the CUDA sources are identical to the upstream baseline, which is
    # what lets one built extension serve both regression checkouts.
    sh(["git", "-C", str(repo), "fetch", "-q", "--depth=1", UPSTREAM, BASELINE_SHA])
    diff = subprocess.check_output(["git", "-C", str(repo), "diff", "--stat", "FETCH_HEAD", "HEAD", "--",
                                    "areno/accel", "setup.py"], text=True)
    (WORK / "accel-diff-vs-baseline.txt").write_text(diff)
    log("accel diff vs baseline: " + (diff.strip() or "<none>"))
    wheels = WORK / "wheels"; wheels.mkdir(exist_ok=True)
    env = {"TORCH_CUDA_ARCH_LIST": "7.5", "MAX_JOBS": "4", "ARENO_BUILD_EXT": "1"}
    sh([sys.executable, "-m", "pip", "wheel", "-q", "--no-build-isolation", "--no-deps", "-w", str(wheels), str(repo)],
       env=env, timeout=3600)
    wheel = next(wheels.glob("areno-*.whl"))
    (wheels / "TORCH_TAG").write_text(torch_tag())
    log(f"built {wheel.name} for {torch_tag()}")
    pip("--no-deps", str(wheel))
    from huggingface_hub import snapshot_download
    raw = Path(snapshot_download(MODEL_ID, cache_dir=str(WORK / "hf-cache")))
    model = fp16_model(raw, WORK / "model-fp16")
    shutil.rmtree(WORK / "hf-cache", ignore_errors=True)
    log(f"model ready at {model}")
    # Source snapshots for the lanes (batched + deterministic) so lanes do not
    # depend on GitHub being reachable.
    for sha, name in ((BATCHED_SHA, "src-batched.tar"), (DETERMINISTIC_SHA, "src-deterministic.tar")):
        d = checkout(sha, SRC.parent / name.replace(".tar", ""))
        with tarfile.open(WORK / name, "w") as tar:
            tar.add(d, arcname=name.replace(".tar", ""))
    # Smoke job: one quality config, sync + lag1, validates fp16 training on T4
    # and gives an L4-comparable point (seed41 tokens64: L4 bf16 got 0.664 / 0.680).
    out = WORK / "smoke"
    sh([sys.executable, "examples/async_policy/tools/matrix.py", "--suite", "quality", "--model-path", str(model),
        "--output-dir", str(out), "--steps", "55", "--warmup", "5", "--backend", "local", "--seeds", "41",
        "--select", "seed41-tokens64-n4-q2-k1-c1-sync", "seed41-tokens64-n4-q2-k1-c1-lag1",
        "--job-timeout-s", "2400", "--campaign-timeout-s", str(int(remaining()) - 600), "--max-attempts", "2",
        "--execute"], cwd=repo, check=False)
    summarize(out)


# ----------------------------------------------------------------------------- lanes
def restore_build():
    assert BUILD_INPUT, "attach the build kernel output as a kernel source"
    install_runtime_deps()
    tag = (BUILD_INPUT / "wheels" / "TORCH_TAG").read_text().strip()
    if tag != torch_tag():
        raise SystemExit(f"wheel built for {tag} but session has {torch_tag()}; rebuild")
    pip("--no-deps", str(next((BUILD_INPUT / "wheels").glob("areno-*.whl"))))
    for name in ("src-batched", "src-deterministic"):
        with tarfile.open(BUILD_INPUT / f"{name}.tar") as tar:
            tar.extractall(SRC.parent)
    model = fp16_model(BUILD_INPUT / "model-fp16", WORK / "model-fp16")
    return SRC.parent / "src-batched", SRC.parent / "src-deterministic", model


def restore_state(campaign: Path):
    """Resume: copy a prior run's campaign directory so matrix.py skips done jobs."""
    if STATE_INPUT and (STATE_INPUT / "campaign").exists():
        shutil.copytree(STATE_INPUT / "campaign", campaign, dirs_exist_ok=True)
        log(f"restored prior campaign state from {STATE_INPUT}")


def matrix(repo, model, campaign, suite, extra):
    if remaining() < 1800:
        log(f"skip {suite}: {int(remaining())}s left"); return
    sh([sys.executable, "examples/async_policy/tools/matrix.py", "--suite", suite, "--model-path", str(model),
        "--output-dir", str(campaign / suite), "--steps", "55", "--warmup", "5", "--backend", "local",
        "--seeds", "41", "42", "43", "--job-timeout-s", "3600",
        "--campaign-timeout-s", str(int(remaining()) - 300), "--max-attempts", "3", *extra, "--execute"],
       cwd=repo, check=False)


def summarize(campaign: Path):
    rows = []
    for attempt in campaign.rglob("attempt.json"):
        a = json.loads(attempt.read_text())
        rows.append({"path": str(attempt.parent.relative_to(campaign)), "status": a.get("status"),
                     "seconds": round((a.get("finished_at") or time.time()) - a["started_at"])})
    for ev in campaign.rglob("evaluation-*.json"):
        try:
            d = json.loads(ev.read_text())
            rows.append({"path": str(ev.relative_to(campaign)), "accuracy": d.get("accuracy"),
                         "mean_response_tokens": d.get("mean_response_tokens")})
        except Exception:
            pass
    (WORK / f"{LANE}-summary.json").write_text(json.dumps(rows, indent=1))
    log(f"summary: {len(rows)} rows -> {LANE}-summary.json")


def role_lane(n):
    gpu_info()
    repo, det_repo, model = restore_build()
    campaign = WORK / "campaign"; campaign.mkdir(exist_ok=True)
    restore_state(campaign)
    if n == 1:
        matrix(repo, model, campaign, "quality", [])
    elif n == 2:
        matrix(repo, model, campaign, "gspo", [])
        matrix(repo, model, campaign, "capacity", ["--select", "*-q2-k1-c1-lag1", "*-q2-k1-c2-lag1",
                                                   "*-q2-k1-c4-lag1", "*-sync-sync"])
    elif n == 3:
        matrix(repo, model, campaign, "batching", [])
        matrix(repo, model, campaign, "throughput", ["--select", "*tokens512*"])
    summarize(campaign)


def role_lane0():
    gpu_info()
    repo, det_repo, model = restore_build()
    out = WORK / "campaign"; out.mkdir(exist_ok=True)
    # T6: strict full-parameter resume, seed 43, on the deterministic branch.
    sh([sys.executable, "examples/async_policy/tools/resume_run.py", "--model-path", str(model),
        "--output-dir", str(out / "resume-full-seed43"), "--full-parameters", "--seed", "43",
        "--deterministic", "--audit-each-step"], cwd=det_repo, check=False, timeout=3600)
    for big in (out / "resume-full-seed43").rglob("checkpoint-*"):
        shutil.rmtree(big, ignore_errors=True)  # keep audits/hashes, drop multi-GB optimizer states
    # T7: regression against the upstream baseline, sharing the built extension.
    baseline = SRC.parent / "areno-upstream"
    if not baseline.exists():
        sh(["git", "init", "-q", str(baseline)])
        sh(["git", "-C", str(baseline), "fetch", "-q", "--depth=1", UPSTREAM, BASELINE_SHA])
        sh(["git", "-C", str(baseline), "checkout", "-q", "--detach", "FETCH_HEAD"])
    import areno.accel as accel
    so = next(Path(accel.__file__).parent.glob("_areno_accel*.so"))
    target = baseline / "areno" / "accel" / so.name
    if not target.exists():
        target.symlink_to(so)
    sh([sys.executable, "tests/async_policy_validation/sdk_regression.py", "--baseline-root", str(baseline),
        "--model-path", str(model), "--data-path", str(repo / "examples/async_policy/smoke_prompts.jsonl"),
        "--output-dir", str(out / "regression")], cwd=repo, check=False, timeout=1800)
    for mode in ("faults", "extended-faults"):
        sh([sys.executable, "examples/async_policy/tools/gpu_run.py", "--mode", mode, "--model-path", str(model),
            "--output-dir", str(out / mode)], cwd=repo, check=False, timeout=1800)
    summarize(out)


if __name__ == "__main__":
    log(f"role={LANE} budget={BUDGET_S}s python={sys.version.split()[0]}")
    try:
        if LANE == "build":
            role_build()
        elif LANE == "lane0":
            role_lane0()
        else:
            role_lane(int(LANE[-1]))
    finally:
        shutil.rmtree(WORK / "model-fp16", ignore_errors=True) if LANE != "build" else None
        log("done")
