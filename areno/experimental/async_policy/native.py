"""Native CUDA adapters for two independent TP=1 / DP=1 worker partitions.

Version ownership and admission remain in DualEngineBridge. This module only
translates model operations to the existing worker protocol and bounds waits.
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict
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
            if handle.done():
                results[index] = handle.result(timeout=0)
                del pending[index]
        if pending:
            next(iter(pending.values())).wait(min(0.01, deadline.remaining() or 0.01))
    return results


class NativeCudaEnginePair:
    """Own native workers; expose the three existing bridge protocols."""

    def __init__(
        self, *, model_path: str, config: CudaConfig, sampling_params: SamplingParams,
        tokenizer: Any, n_samples: int = 4, mini_bs: int = 1, seed: int = 41, worker_cls: type | None = None,
        resume_from: str | None = None, loss_fn: Callable | None = None,
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
        self.resume_from = resume_from
        self.initial_policy_version = 0
        if resume_from is not None:
            from areno.adapters.config import LoraConfig

            metadata = json.loads((Path(resume_from) / "training_state.json").read_text())
            if metadata.get("format") != 1 or metadata.get("has_lora") != (config.lora is not None):
                raise ValueError("incompatible training-state checkpoint")
            self.initial_policy_version = metadata["global_step"]
            if type(self.initial_policy_version) is not int or self.initial_policy_version < 0:
                raise ValueError("checkpoint policy version must be a nonnegative integer")
            if config.lora is not None:
                self.config.lora = LoraConfig(adapter_path=resume_from)
            else:
                self.model_path = resume_from
        self.sampling_params = sampling_params.model_copy(deep=True)
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.mini_bs = mini_bs
        self.seed = seed
        if loss_fn is None:
            from areno.api.algorithms import grpo_loss_fn

            loss_fn = grpo_loss_fn
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable")
        self.loss_fn = loss_fn
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
        from areno import ArenoEngine, OptimizerConfig, RuntimeConfig
        from areno.api.backend.cuda.losses import dispatch_loss
        from areno.api.context import Context
        from areno.api.tokenizer import eos_token_ids
        from areno.engine.protocol import ClusterPartition, DistributedWorldSpec, start_partitioned_clusters

        deadline = Deadline(timeout_s)
        cfg = self.config
        train_part = ClusterPartition("train", 0, 1, 1, tuple(cfg.devices))
        rollout_part = ClusterPartition("rollout", 1, 1, 1, tuple(cfg.rollout_devices))
        world = DistributedWorldSpec("127.0.0.1", 0, 2, train_part, rollout_part)
        self._context = Context(1, self.model_path, self.tokenizer, cfg, eos_token_ids(self.model_path, self.tokenizer))
        for part in (train_part, rollout_part):
            engine = ArenoEngine.from_pretrained(
                self.model_path, start=False,
                base_model_name_or_path=cfg.base_model_name_or_path or self.model_path,
                loss_fn=dispatch_loss if part.role == "train" else None,
                optimizer_config=OptimizerConfig(**cfg.optimizer), runtime_config=RuntimeConfig(**cfg.runtime),
                tp_size=1, dp_size=1, sequence_parallel=cfg.sequence_parallel,
                devices=list(part.devices), role=part.role, lora_config=cfg.lora,
                reference_mode=cfg.reference_mode, policy_sync_bucket_mb=cfg.policy_sync_bucket_mb,
                cluster_kwargs={"world_spec": world, "partition": part},
            )
            # Construction is lazy. Preserve the pair's explicit LoRA seed and
            # allow validation workers to be selected before any process spawns.
            engine.config.lora_seed = self.seed
            if self._worker_cls is not None:
                engine.cluster.worker_cls = self._worker_cls
            self._clusters[part.role] = engine.cluster
        try:
            start_partitioned_clusters(
                self._clusters["train"], self._clusters["rollout"], world,
                timeout_s=deadline.remaining(), stop_event=self._stop_event,
            )
        except InterruptedError as exc:
            if self._stop_event is not None and self._stop_event.is_set():
                raise PipelineClosed("native startup cancelled by pipeline stop") from exc
            raise
        deadline.check()
        self._initialized = True
        if self.resume_from is not None:
            from areno.engine.protocol import Op, SaveCheckpointPayload

            restored = self._call("train", Op.LOAD_TRAINING_STATE, SaveCheckpointPayload(path=self.resume_from), deadline)[0]
            if restored["global_step"] != self.initial_policy_version:
                raise RuntimeError("restored optimizer and coordinator starting versions disagree")

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
        return self.generate_batch((prompt,), version, timeout_s=timeout_s)[0]

    def generate_batch(
        self, prompts: tuple[AsyncPrompt, ...], version: int, *, timeout_s: float | None,
    ) -> dict[int, RolloutResult]:
        """Expand groups to stable worker rows and restore each group by input position."""
        from areno.api.backend.cuda.generation import rollout_options
        from areno.api.models import RolloutResult, RolloutSequence
        from areno.engine.protocol import Op, RolloutPayload

        deadline = Deadline(timeout_s)
        self._require_ready()
        params = self.sampling_params
        if not prompts:
            raise ValueError("a native rollout session requires at least one prompt group")
        for prompt in prompts:
            if prompt.record.get("features") is not None:
                raise ValueError("native async adapters currently accept text-only prompts")
            if params.max_prompt_len is not None and len(prompt.input_tokens) > params.max_prompt_len:
                raise ValueError("prompt exceeds the configured semantic token limit")
        options = rollout_options(self._context, params)
        options["sampling_params"].seed = params.seed if params.seed is not None else self.seed
        rows = [list(prompt.input_tokens) for prompt in prompts for _ in range(self.n_samples)]
        capacity = min(len(rows), self.config.max_running_prompts)
        prompt_capacity = max(max(map(len, rows)), int(options["max_prompt_len"] or 0))
        cache_len = prompt_capacity + params.max_new_tokens
        block_size = self._clusters["rollout"].config.runtime.kv_block_size
        blocks = (cache_len + block_size - 1) // block_size
        payload = RolloutPayload(
            prompts_by_dp=[rows], prompt_indices_by_dp=[list(range(len(rows)))], prompt_features_by_dp=None,
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
        if len(result.response_ids) != len(rows):
            raise RuntimeError("native rollout returned incomplete prompt groups")
        # The TP=1 worker restores state-row order before returning RolloutOutput;
        # completion order and prompt text must never be used as group identity.
        if result.prompt_ids != rows:
            raise RuntimeError("native rollout prompt rows disagree with the requested order")
        sequences = [RolloutSequence(
            resp_tokens=list(tokens), resp_logprobs=result.logprobs[index, :len(tokens)].tolist(),
            routed_experts=None if result.routed_experts is None else result.routed_experts[index],
        ) for index, tokens in enumerate(result.response_ids)]
        return {index: RolloutResult(adapter_version=result.adapter_version,
                                     sequences=sequences[index * self.n_samples:(index + 1) * self.n_samples])
                for index in range(len(prompts))}

    def train(self, batch: BatchEnvelope, version: int, *, timeout_s: float | None) -> bool:
        from areno.api.backend.cuda.training import make_train_pack
        from areno.engine.data import to_cpu
        from areno.engine.protocol import Op, TrainPayload

        deadline = Deadline(timeout_s)
        packs = []
        for start in range(0, len(batch.sequences), self.mini_bs):
            pack = make_train_pack(list(batch.sequences[start:start + self.mini_bs]))
            pack["_loss_fn"] = self.loss_fn
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

    def save_training_checkpoint(self, path: str, *, timeout_s: float | None = 60) -> str:
        """Atomically save policy and optimizer state while holding the train lease."""
        from areno.engine.protocol import Op, SaveCheckpointPayload

        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"checkpoint already exists: {path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        deadline = Deadline(timeout_s)
        try:
            self.save_checkpoint(str(staging), timeout_s=deadline.remaining())
            self._call("train", Op.SAVE_TRAINING_STATE, SaveCheckpointPayload(path=str(staging)), deadline)
            deadline.check()
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return str(destination)

    def close(self, *, timeout_s: float | None) -> None:
        if self._closed:
            return
        deadline = Deadline(timeout_s)
        clusters = list(self._clusters.items())
        self.worker_exits = []
        failures = []
        for _, cluster in clusters:
            try:
                cluster.request_shutdown()
            except Exception as exc:
                failures.append(exc)
        for index, (role, cluster) in enumerate(clusters):
            remaining = deadline.remaining()
            budget = 30.0 if remaining is None else remaining / (len(clusters) - index)
            try:
                exits = cluster.close(timeout_s=budget)
            except Exception as exc:
                failures.append(exc)
                exits = cluster.worker_exits
            self.worker_exits.extend({"role": role, **asdict(row)} for row in exits)
        if failures or any(row["alive"] for row in self.worker_exits):
            raise ShutdownTimeout(f"native cluster shutdown incomplete: {self.worker_exits}") from (
                failures[0] if failures else None
            )
        self._closed = True


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

    def generate_batch(self, prompts, version, *, timeout_s):
        return self.pair.generate_batch(prompts, version, timeout_s=timeout_s)

    def close(self, *, timeout_s):
        # The train endpoint owns both process clusters and their TCPStore.
        pass


class _NativeSync:
    def __init__(self, pair: NativeCudaEnginePair):
        self.pair = pair

    def transfer(self, plan, *, timeout_s):
        self.pair.transfer(plan, timeout_s=timeout_s)
