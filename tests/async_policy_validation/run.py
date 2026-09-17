"""Run the opt-in CPU audit; preserve failures and exit nonzero on violations."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def run_group(node: str, output: Path, root: Path, timeout: float) -> dict:
    name = node.rsplit("::", 1)[-1] if "::" in node else "existing_suite"
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", "-q", "--tb=short", "-o", "faulthandler_timeout=15",
               f"--junitxml={directory / 'junit.xml'}", *node.split()]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="", ARENO_AUDIT_ARTIFACT_DIR=str(directory))
    start = time.monotonic()
    with (directory / "output.log").open("w") as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
    cases = []
    xml = directory / "junit.xml"
    if xml.exists():
        for case in ET.parse(xml).iter("testcase"):
            failure = case.find("failure")
            error = case.find("error")
            skipped = case.find("skipped")
            bad = failure if failure is not None else error
            cases.append({"name": case.attrib["name"], "status": "fail" if bad is not None else
                          "skip" if skipped is not None else "pass",
                          "detail": bad.text if bad is not None else ""})
    result = {"node": node, "command": command, "exit_code": code, "timeout": timed_out,
              "elapsed_s": time.monotonic() - start, "cases": cases, "artifact_dir": str(directory)}
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", choices=("all", "protocol", "torch"), default="all")
    parser.add_argument("--case", help="Only functions whose name contains this substring")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = ["known_cases.py", "protocol_cases.py", "torch_cases.py"]
    if args.layer == "protocol":
        files.remove("torch_cases.py")
    elif args.layer == "torch":
        files = ["torch_cases.py"]
    nodes = []
    for filename in files:
        path = Path(__file__).parent / filename
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                if args.case is None or args.case in node.name:
                    nodes.append(f"{path.relative_to(root)}::{node.name}")
    if not nodes:
        parser.error("no matching cases")
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": "",
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "pydantic", "pytest")},
        "nodes": nodes, "jobs": args.jobs, "timeout_s": args.timeout,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run_group, node, output, root, args.timeout) for node in nodes]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{'PASS' if result['exit_code'] == 0 else 'FAIL'} {result['node']} "
                  f"({result['elapsed_s']:.2f}s)", flush=True)
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    counts = {status: sum(case["status"] == status for r in results for case in r["cases"])
              for status in ("pass", "fail", "skip")}
    counts["timeout"] = sum(r["timeout"] for r in results)
    counts["failed_groups"] = sum(r["exit_code"] != 0 for r in results)
    (output / "summary.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts), flush=True)
    return 1 if counts["failed_groups"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
