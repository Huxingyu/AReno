"""Durable experiment evidence and conservative, shared Modal spending bounds."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# Public function-container prices, checked at https://modal.com/pricing.
# Wall time includes provisioning/waiting and therefore overestimates compute.
PRICES = {"date": "2026-09-16", "l4_per_s": 0.000222,
          "cpu_core_per_s": 0.0000131, "memory_gib_per_s": 0.00000222}


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    """Publish complete JSON atomically; fsync it before publishing the name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def exclusive_lock(path: Path):
    """Only one local controller may mutate a campaign or shared budget."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another controller owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def file_manifest(root: Path, paths=None) -> dict:
    paths = paths if paths is not None else sorted(path for path in root.rglob("*") if path.is_file())
    files = {str(path.relative_to(root)): {"size": path.stat().st_size, "sha256": sha256_file(path)}
             for path in paths}
    if not files:
        raise ValueError(f"empty evidence directory: {root}")
    return {"files": files, "sha256": fingerprint(files)}


def verify_manifest(root: Path, manifest: dict) -> None:
    if not manifest.get("files") or fingerprint(manifest["files"]) != manifest.get("sha256"):
        raise ValueError("invalid evidence manifest")
    for name, expected in manifest["files"].items():
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("evidence path escapes its directory")
        if not path.is_file() or path.stat().st_size != expected["size"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"missing or changed evidence: {path}")


def complete_directory(root: Path, identity: dict, *, paths=None) -> None:
    """The completion marker is last and never hashes itself."""
    if paths is None:
        paths = sorted(path for path in root.rglob("*") if path.is_file()
                       and path.name not in {"complete.json", "console.log", "modal-task.json"})
    write_json(root / "complete.json", {"identity": identity, "manifest": file_manifest(root, paths)})


def verify_complete(root: Path, identity: dict | None = None) -> dict:
    marker = json.loads((root / "complete.json").read_text())
    if identity is not None and marker["identity"] != identity:
        raise ValueError("completed task settings changed")
    verify_manifest(root, marker["manifest"])
    return marker


def code_fingerprint(*paths: str, revision: str = "HEAD") -> str:
    """Hash Git blob IDs, preserving provenance across tool-only commits."""
    tree = subprocess.check_output(["git", "ls-tree", "-r", revision, "--", *paths], cwd=ROOT)
    if not tree:
        raise ValueError("no source files matched the requested fingerprint")
    return hashlib.sha256(tree).hexdigest()


def training_code_fingerprint(revision: str = "HEAD") -> str:
    return code_fingerprint("areno", "examples/async_policy/train.py",
                            "examples/async_policy/tools/gpu_worker.py", revision=revision)


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    """Reserve an entire bounded attempt before any paid Modal work starts.

    An interrupted controller retains the full reservation. Release unused
    funds only after its app is confirmed stopped. This is a conservative
    estimate, not a substitute for Modal's final account invoice.
    """

    def __init__(self, path: Path, limit_usd: float | None = None, *, reserve_usd: float = 1.0):
        self.path = path
        with exclusive_lock(path.with_suffix(".lock")):
            if path.exists():
                state = self.read()
                if limit_usd is not None and state["limit_usd"] != limit_usd:
                    raise ValueError("existing budget limit cannot be changed implicitly")
                if state["prices"] != PRICES:
                    raise ValueError("budget pricing changed; reconcile the existing ledger first")
            else:
                if limit_usd is None or not math.isfinite(limit_usd) or not 0 <= reserve_usd < limit_usd:
                    raise ValueError("a new budget needs a finite limit greater than its safety reserve")
                write_json(path, {"limit_usd": limit_usd, "reserve_usd": reserve_usd,
                                  "prices": PRICES, "created_at": time.time(), "attempts": {}})

    def read(self) -> dict:
        return json.loads(self.path.read_text())

    @staticmethod
    def rate(gpus: int, *, cpu: int = 4, memory_gib: int = 16) -> float:
        if gpus not in (0, 1, 2):
            raise ValueError("only CPU, single-L4 and dual-L4 jobs are budgeted")
        return gpus * PRICES["l4_per_s"] + cpu * PRICES["cpu_core_per_s"] + memory_gib * PRICES["memory_gib_per_s"]

    def reserve(self, attempt_id: str, seconds: float, gpus: int) -> dict:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("attempt duration must be finite and positive")
        with exclusive_lock(self.path.with_suffix(".lock")):
            state = self.read()
            if attempt_id in state["attempts"]:
                raise ValueError("attempt ID already reserved; inspect its previous outcome")
            rate = self.rate(gpus)
            maximum = math.ceil((seconds + 30) * rate * 1_000_000) / 1_000_000
            charged = sum(row["charged_usd"] for row in state["attempts"].values())
            if charged + maximum > state["limit_usd"] - state["reserve_usd"]:
                raise BudgetExceeded(f"budget exhausted: ${charged:.4f} reserved/spent, "
                                     f"${maximum:.4f} needed, ${state['reserve_usd']:.2f} safety reserve")
            entry = {"started_at": time.time(), "seconds_limit": seconds, "gpus": gpus,
                     "rate_usd_per_s": rate, "reserved_usd": maximum, "charged_usd": maximum,
                     "status": "reserved", "confirmed_stopped": False}
            state["attempts"][attempt_id] = entry
            write_json(self.path, state)
            return entry

    def finish(self, attempt_id: str, elapsed_s: float, *, confirmed_stopped: bool, status: str) -> None:
        with exclusive_lock(self.path.with_suffix(".lock")):
            state = self.read()
            entry = state["attempts"][attempt_id]
            entry.update(elapsed_s=elapsed_s, finished_at=time.time(), status=status,
                         confirmed_stopped=confirmed_stopped)
            if confirmed_stopped:
                entry["charged_usd"] = math.ceil((elapsed_s + 30) * entry["rate_usd_per_s"] * 1_000_000) / 1_000_000
            write_json(self.path, state)

    def summary(self) -> dict:
        state = self.read()
        charged = sum(row["charged_usd"] for row in state["attempts"].values())
        return {"limit_usd": state["limit_usd"], "safety_reserve_usd": state["reserve_usd"],
                "conservative_charged_usd": charged,
                "available_usd": max(0.0, state["limit_usd"] - state["reserve_usd"] - charged),
                "unconfirmed_attempts": [key for key, row in state["attempts"].items()
                                         if not row["confirmed_stopped"]]}
