"""Native CUDA adapters for two independent TP=1 / DP=1 worker partitions.

Version ownership and admission remain in DualEngineBridge. This module only
translates model operations to the existing worker protocol and bounds waits.
"""

from __future__ import annotations

import copy
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .contracts import AsyncPrompt, BatchEnvelope, BridgeStateError, PipelineClosed, ShutdownTimeout, SyncPlan
from .coordination import Deadline

if TYPE_CHECKING:
    from areno.api.config import CudaConfig
    from areno.api.models import RolloutResult, SamplingParams


def _wait_handles(handles: list, deadline: Deadline, stop_event: threading.Event | None = None) -> list:
    """Observe either side's error promptly, including paired NCCL calls."""
    pending = dict(enumerate(handles))
    results = [None] * len(handles)
    while pending:
        if stop_event is not None and stop_event.is_set():
            raise PipelineClosed("native RPC cancelled by pipeline stop")
        deadline.check()
        for index, handle in tuple(pending.items()):
            # ClusterCallHandle has no public readiness probe. Waiting on its
            # existing event avoids adding threads that survive a failed peer.
            if handle._pending.event.is_set():
                results[index] = handle.result(timeout=0)
                del pending[index]
        if pending:
            next(iter(pending.values()))._pending.event.wait(min(0.01, deadline.remaining() or 0.01))
    return results


def _make_cluster(config, worker_cls, world_spec, partition, deadline: Deadline, stop_event):
    from areno.engine.protocol import TPCluster

    class BoundedStartupCluster(TPCluster):
        def _wait_for_worker_ready(self, pending: set[int]) -> None:
            while pending:
                if stop_event is not None and stop_event.is_set():
                    raise PipelineClosed("native startup cancelled by pipeline stop")
                deadline.check()
                try:
                    rank, result = self.result_queue.get(timeout=min(0.1, deadline.remaining() or 0.1))
                except queue.Empty:
                    dead = self._dead_pending_workers(pending)
                    if dead:
                        raise RuntimeError(f"native workers exited during startup: {dead}") from None
                    continue
                if not result.ok:
                    raise RuntimeError(f"rank {rank} failed during startup:\n{result.error}")
                pending.discard(rank)

        def _abort_start(self) -> None:
            # Keep process/queue references so the bridge's one cleanup budget
            # can reap them and report any survivor after partial startup.
            for process in self.processes:
                if process.is_alive():
                    process.terminate()

    return BoundedStartupCluster(config, worker_cls, world_spec=world_spec, partition=partition)


class NativeCudaEnginePair:
    """Own native workers; expose the three existing bridge protocols."""

    def __init__(
        self, *, model_path: str, config: CudaConfig, sampling_params: SamplingParams,
        tokenizer: Any, n_samples: int = 4, mini_bs: int = 1, seed: int = 41, worker_cls: type | None = None,
    ):
        if not Path(model_path).is_dir():
            raise ValueError("native async adapters require a resolved local checkpoint (use ModelScope first)")
        if (config.tp_size, config.dp_size) not in ((1, None), (1, 1)):
            raise ValueError("native async adapters currently require train TP=1 / DP=1")
        if config.resolved_rollout_tp_size() != 1 or len(config.devices or ()) != 1 or len(config.rollout_devices or ()) != 1:
            raise ValueError("native async adapters require one train GPU and one rollout GPU")
        if set(config.devices).intersection(config.rollout_devices):
            raise ValueError("native async adapters require disjoint physical GPU indices")
        if n_samples < 1 or mini_bs < 1 or config.max_running_prompts < 1:
            raise ValueError("sample, microbatch, and rollout capacity must be positive")
        if config.dummy_load:
            raise ValueError("native async acceptance requires actual checkpoint weights")
        self.model_path = model_path
        self.config = copy.deepcopy(config)
        self.sampling_params = sampling_params.model_copy(deep=True)
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.mini_bs = mini_bs
        self.seed = seed
        self._worker_cls = worker_cls
        self._clusters: dict[str, Any] = {}
        self._initialized = False
        self._closed = False
        self._context = None
        self._stop_event: threading.Event | None = None
        self.train_stats: list[dict] = []
        self.sync_stats: list[dict] = []
        self.worker_exits: list[dict] = []
        self.train_engine = _NativeTrain(self)
        self.rollout_engine = _NativeRollout(self)
        self.weight_sync = _NativeSync(self)

    def bind_stop_event(self, event: threading.Event) -> None:
        if self._stop_event is not None and self._stop_event is not event:
            raise BridgeStateError("native pair cannot be shared by different supervisors")
        self._stop_event = event

    def initialize(self, *, timeout_s: float | None) -> None:
        if self._initialized:
            return
        if self._closed or self._clusters:
            raise BridgeStateError("native pair is single-use")
        from areno.api.backend.cuda.losses import dispatch_loss
        from areno.api.context import Context
        from areno.api.tokenizer import eos_token_ids
        from areno.engine.config import EngineConfig, OptimizerConfig, RuntimeConfig
        from areno.engine.protocol import ClusterPartition, DistributedWorldSpec, start_partitioned_clusters
        from areno.engine.worker import ArenoWorker
        from areno.models.registry import config_from_hf

        deadline = Deadline(timeout_s)
        cfg = self.config
        train_part = ClusterPartition("train", 0, 1, 1, tuple(cfg.devices))
        rollout_part = ClusterPartition("rollout", 1, 1, 1, tuple(cfg.rollout_devices))
        world = DistributedWorldSpec("127.0.0.1", 0, 2, train_part, rollout_part)
        model = config_from_hf(self.model_path)
        self._context = Context(1, self.model_path, self.tokenizer, cfg, eos_token_ids(self.model_path, self.tokenizer))
        for part in (train_part, rollout_part):
            engine_config = EngineConfig(
                model=copy.deepcopy(model), model_path=self.model_path,
                base_model_name_or_path=cfg.base_model_name_or_path or self.model_path,
                train_loss_fn=dispatch_loss if part.role == "train" else None,
                optimizer=OptimizerConfig(**cfg.optimizer), runtime=RuntimeConfig(**cfg.runtime),
                tp_size=1, dp_size=1, sequence_parallel=cfg.sequence_parallel,
                devices=list(part.devices), role=part.role, lora=cfg.lora, lora_seed=self.seed,
                reference_mode=cfg.reference_mode, policy_sync_bucket_mb=cfg.policy_sync_bucket_mb,
            )
            self._clusters[part.role] = _make_cluster(
                engine_config, self._worker_cls or ArenoWorker, world, part, deadline, self._stop_event,
            )
        start_partitioned_clusters(self._clusters["train"], self._clusters["rollout"], world)
        deadline.check()
        self._initialized = True

    def _require_ready(self) -> None:
        if not self._initialized or self._closed:
            raise BridgeStateError("native pair is not initialized or already closed")

    def _call(self, role: str, op, payload, deadline: Deadline):
        self._require_ready()
        deadline.check()
        if self._stop_event is not None and self._stop_event.is_set():
            raise PipelineClosed("native RPC cancelled by pipeline stop")
        return _wait_handles([self._clusters[role].submit(op, payload)], deadline, self._stop_event)[0]

    def generate(self, prompt: AsyncPrompt, version: int, *, timeout_s: float | None) -> RolloutResult:
        from areno.api.backend.cuda.generation import rollout_options
        from areno.api.models import RolloutResult, RolloutSequence
        from areno.engine.protocol import Op, RolloutPayload

        deadline = Deadline(timeout_s)
        self._require_ready()
        params = self.sampling_params
        if prompt.record.get("features") is not None:
            raise ValueError("native async adapters currently accept text-only prompts")
        if params.max_prompt_len is not None and len(prompt.input_tokens) > params.max_prompt_len:
            raise ValueError("prompt exceeds the configured semantic token limit")
        options = rollout_options(self._context, params)
        options["sampling_params"].seed = params.seed if params.seed is not None else self.seed
        capacity = min(self.n_samples, self.config.max_running_prompts)
        prompt_capacity = max(len(prompt.input_tokens), int(options["max_prompt_len"] or 0))
        cache_len = prompt_capacity + params.max_new_tokens
        block_size = self._clusters["rollout"].config.runtime.kv_block_size
        blocks = (cache_len + block_size - 1) // block_size
        rows = [list(prompt.input_tokens) for _ in range(self.n_samples)]
        payload = RolloutPayload(
            prompts_by_dp=[rows], prompt_indices_by_dp=[list(range(self.n_samples))], prompt_features_by_dp=None,
            max_new_tokens=params.max_new_tokens, eos_token_id=options["eos_token_id"],
            sampling_params=options["sampling_params"], max_running_seqs=capacity, max_cache_len=cache_len,
            max_blocks_per_seq=blocks, max_prefill_tokens=capacity * prompt_capacity,
            num_blocks=capacity * blocks, block_size=block_size,
        )
        self._call("rollout", Op.ROLLOUT_SESSION_BEGIN, None, deadline)
        result = self._call("rollout", Op.INFER_ROLLOUT, payload, deadline)[0]
        self._call("rollout", Op.ROLLOUT_SESSION_END, None, deadline)
        if result.adapter_version is not None and result.adapter_version != version:
            raise RuntimeError("native rollout weights disagree with bridge version")
        if len(result.response_ids) != self.n_samples:
            raise RuntimeError("native rollout returned an incomplete prompt group")
        return RolloutResult(adapter_version=result.adapter_version, sequences=[
            RolloutSequence(
                resp_tokens=list(tokens), resp_logprobs=result.logprobs[index, :len(tokens)].tolist(),
                routed_experts=None if result.routed_experts is None else result.routed_experts[index],
            ) for index, tokens in enumerate(result.response_ids)
        ])

    def train(self, batch: BatchEnvelope, version: int, *, timeout_s: float | None) -> bool:
        from areno.api.algorithms import grpo_loss_fn
        from areno.api.backend.cuda.training import make_train_pack
        from areno.engine.data import to_cpu
        from areno.engine.protocol import Op, TrainPayload

        deadline = Deadline(timeout_s)
        packs = []
        for start in range(0, len(batch.sequences), self.mini_bs):
            pack = make_train_pack(list(batch.sequences[start:start + self.mini_bs]))
            pack["_loss_fn"] = grpo_loss_fn
            packs.append([pack])
        payload = TrainPayload(to_cpu(packs, share_memory=True), gradient_accumulation_steps=len(packs))
        stats = self._call("train", Op.TRAIN, payload, deadline)[0]
        if len(stats) != len(packs) or any(type(row["stepped"]) is not bool for row in stats):
            raise RuntimeError("native training did not return explicit optimizer-step results")
        updates = sum(row["stepped"] for row in stats)
        if updates > 1:
            raise RuntimeError("a native async batch performed more than one optimizer update")
        for row in stats:
            self.train_stats.append({"batch_id": batch.batch_id, "version_before": version, **row})
        if stats and int(stats[-1]["global_step"]) != version + updates:
            raise RuntimeError("native optimizer step disagrees with bridge version")
        return bool(updates)

    def transfer(self, plan: SyncPlan, *, timeout_s: float | None) -> None:
        from areno.engine.protocol import Op, PolicySyncPayload

        self._require_ready()
        deadline = Deadline(timeout_s)
        clusters = (self._clusters["train"], self._clusters["rollout"])
        plans = _wait_handles([cluster.submit(Op.POLICY_SYNC_PLAN) for cluster in clusters], deadline, self._stop_event)
        if not plans[0][0] or plans[0] != plans[1]:
            raise RuntimeError("native policy synchronization layouts are empty or unequal")
        payload = PolicySyncPayload(version=plan.target_version, bucket_bytes=self.config.policy_sync_bucket_mb * 1024**2)
        results = _wait_handles([
            clusters[0].submit(Op.POLICY_SYNC_PUBLISH, payload),
            clusters[1].submit(Op.POLICY_SYNC_RECEIVE, payload),
        ], deadline, self._stop_event)
        summaries = [result[0] for result in results]
        if any(row["version"] != plan.target_version or row["bytes"] <= 0 for row in summaries):
            raise RuntimeError("native policy transfer was not acknowledged by both workers")
        self.sync_stats.append({"source_version": plan.source_version, "target_version": plan.target_version,
                                "train": summaries[0], "rollout": summaries[1]})

    def save_checkpoint(self, path: str, *, role: str = "train", timeout_s: float | None = 60) -> str:
        """Caller must hold the bridge lease or exclusive synchronization slot."""
        from areno.engine.protocol import ExportAdapterPayload, Op, SaveCheckpointPayload

        if self.config.lora is not None:
            op, payload = Op.EXPORT_ADAPTER, ExportAdapterPayload(path=path)
        else:
            op, payload = Op.SAVE_CHECKPOINT, SaveCheckpointPayload(path=path)
        result = self._call(role, op, payload, Deadline(timeout_s))[0]
        return result["path"]

    def close(self, *, timeout_s: float | None) -> None:
        from areno.engine.protocol import Command, Op

        if self._closed:
            return
        deadline = Deadline(timeout_s)
        clusters = list(self._clusters.values())
        processes = [process for cluster in clusters for process in cluster.processes]
        for cluster in clusters:
            cluster._pump_stop.set()
            for commands in cluster.cmd_queues:
                commands.put(Command(op=Op.SHUTDOWN))
        remaining = deadline.remaining()
        grace = Deadline(2.0 if remaining is None else min(2.0, remaining / 3))
        for process in processes:
            process.join(timeout=grace.remaining())
        for process in processes:
            if process.is_alive():
                process.terminate()
        remaining = deadline.remaining()
        terminate_wait = Deadline(2.0 if remaining is None else min(2.0, remaining / 2))
        for process in processes:
            process.join(timeout=terminate_wait.remaining())
        for process in processes:
            if process.is_alive():
                process.kill()
        for process in processes:
            process.join(timeout=deadline.remaining())
        self.worker_exits = [{"pid": process.pid, "exitcode": process.exitcode, "alive": process.is_alive()}
                             for process in processes]
        if any(row["alive"] for row in self.worker_exits):
            raise ShutdownTimeout(f"native workers still alive: {self.worker_exits}")
        for cluster in clusters:
            if cluster._pump_thread is not None:
                cluster._pump_thread.join(timeout=deadline.remaining())
                if cluster._pump_thread.is_alive():
                    raise ShutdownTimeout("native result pump still alive")
                cluster._pump_thread = None
            for channel in [*cluster.cmd_queues, cluster.result_queue]:
                channel.cancel_join_thread()
                channel.close()
            with cluster._pending_lock:
                pending_calls = tuple(cluster._pending_calls.items())
            for request_id, pending in pending_calls:
                cluster._finish_pending_call(request_id, pending, BridgeStateError("native worker pair closed"))
            cluster.started = False
            cluster._rendezvous_store = None
            for process in cluster.processes:
                process.close()
            cluster.processes.clear()
        self._closed = True
        deadline.check()


class _NativeTrain:
    def __init__(self, pair: NativeCudaEnginePair):
        self.pair = pair

    def initialize(self, *, timeout_s):
        self.pair.initialize(timeout_s=timeout_s)

    def bind_stop_event(self, event):
        self.pair.bind_stop_event(event)

    def train(self, batch, version, *, timeout_s):
        return self.pair.train(batch, version, timeout_s=timeout_s)

    def close(self, *, timeout_s):
        self.pair.close(timeout_s=timeout_s)


class _NativeRollout:
    def __init__(self, pair: NativeCudaEnginePair):
        self.pair = pair

    def initialize(self, *, timeout_s):
        self.pair._require_ready()

    def bind_stop_event(self, event):
        self.pair.bind_stop_event(event)

    def generate(self, prompt, version, *, timeout_s):
        return self.pair.generate(prompt, version, timeout_s=timeout_s)

    def close(self, *, timeout_s):
        # The train endpoint owns both process clusters and their TCPStore.
        pass


class _NativeSync:
    def __init__(self, pair: NativeCudaEnginePair):
        self.pair = pair

    def transfer(self, plan, *, timeout_s):
        self.pair.transfer(plan, timeout_s=timeout_s)
