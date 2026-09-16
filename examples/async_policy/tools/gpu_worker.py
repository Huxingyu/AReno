"""Test-only native worker instrumentation and deterministic fault injection."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from areno.engine.inference import InferenceManager
from areno.engine.protocol import Op
from areno.engine.worker import ArenoWorker


class ObservedInference(InferenceManager):
    """Time prefill/decode without adding a synchronization to each token."""

    def _measure(self, stage, operation, input_tokens=0):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        result = operation()
        end.record()
        tokens = result[0] if isinstance(result, tuple) else result
        self.worker._inference_events.append({
            "stage": stage, "begin": begin, "end": end, "cpu_s": time.perf_counter() - started,
            "sampled_tokens": tokens.numel() if tokens is not None else 0, "input_tokens": input_tokens,
        })
        return result

    def _infer_next_token_tensor(self, payload):
        return self._measure("prefill", lambda: super(ObservedInference, self)._infer_next_token_tensor(payload),
                             payload.input_ids.numel())

    def _run_prefill_payload(self, payload):
        return self._measure("prefill", lambda: super(ObservedInference, self)._run_prefill_payload(payload),
                             payload.input_ids.numel())

    def _infer_decode_next_token_tensor(self, *args, **kwargs):
        return self._measure("decode", lambda: super(ObservedInference, self)._infer_decode_next_token_tensor(*args, **kwargs))


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
        self._inference_events = []
        self._rollout_output_metrics = {}
        self.inference = ObservedInference(self)
        self._observation_dir = Path(os.environ["ARENO_R4_OUTPUT"])
        self._observed_counts: dict[str, int] = {}

    def _observe(self, kind, function):
        count = self._observed_counts.get(kind, 0) + 1
        self._observed_counts[kind] = count
        row = {"kind": kind, "ordinal": count, "role": self.config.role,
               "pid": os.getpid(), "device": self.device.index, "start_ns": time.time_ns()}
        profiler = None
        marker = f"areno-clock-{self.config.role}-{kind}-{count}"
        if kind == "rollout":
            self._inference_events = []
            self._rollout_output_metrics = {}
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
            if kind == "rollout":
                stages = {}
                for event in self._inference_events:
                    stage = stages.setdefault(event["stage"], {
                        "calls": 0, "cpu_s": 0.0, "cuda_s": 0.0, "sampled_tokens": 0, "input_tokens": 0,
                    })
                    stage["calls"] += 1
                    for key in ("cpu_s", "sampled_tokens", "input_tokens"):
                        stage[key] += event[key]
                    stage["cuda_s"] += event["begin"].elapsed_time(event["end"]) / 1000
                row["inference"] = {"stages": stages, **self._rollout_output_metrics,
                                    "timing": "CUDA events; synchronized once at rollout completion"}
                self._inference_events = []
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

    def infer_rollout(self, *args, **kwargs):
        result = super().infer_rollout(*args, **kwargs)
        if result is not None:
            self._rollout_output_metrics = {
                "returned_response_tokens": sum(len(tokens) for tokens in result.response_ids),
                "response_rows": len(result.response_ids), "engine_metrics": result.metrics,
            }
        return result

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
        torch.manual_seed(41)
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
        return result
