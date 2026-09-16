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
        from areno.engine.runtime.decode_graph import DecodeGraph

        self._graph_replays = 0
        replay = DecodeGraph.replay_tensors

        def observed_replay(graph, *args, **kwargs):
            result = replay(graph, *args, **kwargs)
            self._graph_replays += 1
            return result

        DecodeGraph.replay_tensors = observed_replay
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
            fault = os.environ.get("ARENO_R4_FAULT", "")
            if kind == "rollout" and fault in {"sigint", "sigterm"} and (self._observation_dir / "checkpoint-1").is_dir():
                (self._observation_dir / "fault-ready.json").write_text(json.dumps({
                    "fault": fault, "pid": os.getpid(), "monotonic_s": time.monotonic(),
                }))
                time.sleep(120)
            result = function()
            torch.cuda.synchronize(self.device)
            row["max_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
            row["max_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
            row["graph_replays"] = self._graph_replays
            row["decode_graph_buckets"] = sorted(self._decode_graphs)
            if self.config.runtime.compile_model:
                from torch._dynamo.utils import counters

                row["compiled_graphs"] = counters["stats"]["unique_graphs"]
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
        if payload.version == 1 and os.environ.get("ARENO_R4_FAULT") == "timeout":
            (self._observation_dir / "fault-ready.json").write_text(json.dumps({
                "fault": "timeout", "pid": os.getpid(), "monotonic_s": time.monotonic(),
            }))
            time.sleep(120)
        if payload.version == 1 and os.environ.get("ARENO_R4_FAULT") == "receive":
            raise RuntimeError("injected native policy receiver failure at version 1")
        return super().receive_policy(payload)


class ResumeAuditWorker(ObservedWorker):
    """Hash the live checkpoint boundary before any resumed training occurs."""

    def __init__(self, config):
        torch.manual_seed(int(os.environ.get("ARENO_RESUME_SEED", "41")))
        if os.environ.get("ARENO_RESUME_DETERMINISTIC") == "1":
            torch.use_deterministic_algorithms(True)
        super().__init__(config)

    def _audit_resume(self, operation):
        from areno.engine.modeling import unwrap_model
        from examples.async_policy.tools.resume_run import state_fingerprints

        state = {"global_step": self._global_step,
                 "model": state_fingerprints(unwrap_model(self.model).state_dict()),
                 "cpu_rng": state_fingerprints(torch.get_rng_state()),
                 "cuda_rng": state_fingerprints(torch.cuda.get_rng_state(self.device)),
                 "optimizer": state_fingerprints(self.optimizer.state_dict())}
        path = self._observation_dir / f"resume-audit-{operation}-{self._global_step}.json"
        path.write_text(json.dumps(state, indent=2) + "\n")

    def handle(self, command):
        if command.op is Op.TRAIN and self._global_step == 0:
            self._audit_resume("before_train")
        result = super().handle(command)
        if self.config.role == "train" and command.op in {Op.SAVE_CHECKPOINT, Op.LOAD_TRAINING_STATE}:
            self._audit_resume(command.op.name.lower())
        if (self.config.role == "train" and command.op is Op.TRAIN
                and os.environ.get("ARENO_RESUME_AUDIT_EACH_STEP") == "1"):
            self._audit_resume("after_train")
        return result
