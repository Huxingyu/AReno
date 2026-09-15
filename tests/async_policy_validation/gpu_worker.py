"""Test-only native worker instrumentation and deterministic fault injection."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from areno.engine.protocol import Op
from areno.engine.worker import ArenoWorker


class ObservedWorker(ArenoWorker):
    def __init__(self, config):
        super().__init__(config)
        self._observation_dir = Path(os.environ["ARENO_R4_OUTPUT"])
        self._observed_counts: dict[str, int] = {}

    def _observe(self, kind, function):
        count = self._observed_counts.get(kind, 0) + 1
        self._observed_counts[kind] = count
        row = {"kind": kind, "ordinal": count, "role": self.config.role,
               "pid": os.getpid(), "device": self.device.index, "start_ns": time.time_ns()}
        profiler = None
        marker = f"areno-clock-{self.config.role}-{kind}-{count}"
        try:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            if os.environ.get("ARENO_R4_PROFILE") == "1" and kind in {"train", "rollout"} and count >= 3:
                profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA])
                profiler.__enter__()
                row["clock_before_ns"] = time.time_ns()
                with torch.profiler.record_function(marker):
                    row["clock_inside_ns"] = time.time_ns()
            if kind == "train" and count == 2 and os.environ.get("ARENO_R4_FAULT") == "worker_exit":
                os._exit(23)
            result = function()
            torch.cuda.synchronize(self.device)
            row["max_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
            row["max_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
            row["ok"] = True
            return result
        except BaseException as exc:
            row["ok"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row["end_ns"] = time.time_ns()
            if profiler is not None:
                profiler.__exit__(None, None, None)
                trace = self._observation_dir / f"trace-{self.config.role}-{kind}-{count}.json"
                profiler.export_chrome_trace(str(trace))
                row["trace"] = trace.name
                row["clock_marker"] = marker
            with (self._observation_dir / f"worker-{self.config.role}.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    def handle(self, command):
        if command.op in {Op.TRAIN, Op.POLICY_SYNC_PUBLISH, Op.POLICY_SYNC_RECEIVE}:
            kind = "train" if command.op == Op.TRAIN else "sync"
            return self._observe(kind, lambda: super(ObservedWorker, self).handle(command))
        return super().handle(command)

    def run_rollout_command(self, command):
        return self._observe("rollout", lambda: super(ObservedWorker, self).run_rollout_command(command))

    def receive_policy(self, payload):
        if payload.version == 1 and os.environ.get("ARENO_R4_FAULT") == "receive":
            raise RuntimeError("injected native policy receiver failure at version 1")
        return super().receive_policy(payload)
