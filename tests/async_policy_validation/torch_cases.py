"""Real CPU parameter tests, plus numerical and ownership contract probes."""

from __future__ import annotations

import threading
from dataclasses import asdict

import pytest
import torch

from areno.api.backend.cuda.losses import grpo_loss_fn
from areno.api.backend.cuda.training import make_train_pack
from areno.engine.runtime.train_step import _pack_train_data
from areno.experimental.async_policy import (
    AsyncPolicyPipeline,
    AsyncPrompt,
    BatchContractError,
    BridgeStateError,
    DualEngineBridge,
    PipelineClosed,
    PolicyPipelineCoordinator,
    build_batch_envelope,
)
from tests.async_policy_validation.protocol_cases import assert_drained, config, save
from tests.async_policy_validation.tiny_backend import (
    TinyPolicy,
    TorchSync,
    engines,
    fingerprint,
    reference_values,
    vector,
)


def prompts(count):
    return [AsyncPrompt(prompt_id=str(i), input_tokens=(1, 3 + i % 10, 4)) for i in range(count)]


def ready_batch(rollout, prompt=None):
    prompt = prompt or prompts(1)[0]
    result = rollout.generate(prompt, rollout.version, timeout_s=2)
    return build_batch_envelope(run_id="tensor-audit", batch_id=f"b{prompt.prompt_id}", epoch=0,
                                policy_version=rollout.version, prompt_items=[prompt],
                                rollout_results=[result], rewards=rollout.records[prompt.prompt_id]["rewards"],
                                eos_token_id=2)


@pytest.mark.parametrize("q,k,cadence,lag", [(1, 1, 1, 1), (2, 2, 4, 2), (4, 4, 4, 0), (1, 4, 1, 4)])
def test_real_cpu_pipeline_versions_gradients_and_sync(q, k, cadence, lag):
    coordinator = PolicyPipelineCoordinator()
    rollout, train, sync, audit = engines(coordinator, skip_calls=(1, 5))
    rollout_started, train_started = threading.Event(), threading.Event()
    rollout.started_events[1] = rollout_started
    rollout.gates[1] = train_started
    train.started_events[0] = train_started
    train.gates[0] = rollout_started
    pipe = AsyncPolicyPipeline(config=config(queue_capacity=q, max_inflight_rollouts=k,
                               weight_sync_interval_updates=cadence, max_policy_lag=lag),
                               data_source=prompts(24), rollout_engine=rollout, train_engine=train,
                               weight_sync=sync, reward_fn=rollout.reward, coordinator=coordinator)
    report = pipe.run()
    assert_drained(pipe)
    assert report.steps == train.updates == coordinator.train_policy_version
    assert report.steps > 2
    assert report.batches_seen == report.produced_batches == 24
    assert len(train.records) + report.batches_dropped_stale == 24
    assert any(not record["stepped"] for record in train.records)
    assert any(record["stepped"] and record["delta"] > 0 for record in train.records)
    assert rollout.version == coordinator.rollout_policy_version
    assert fingerprint(rollout.model) == train.snapshots[rollout.version]
    assert sync.transfers
    active = {"train": 0, "rollout": 0, "sync": 0}
    overlap = False
    for event in audit.events:
        stage = event["stage"]
        active[stage] += 1 if event["action"] == "start" else -1
        assert active[stage] >= 0
        assert not (active["sync"] and (active["train"] or active["rollout"]))
        overlap |= bool(active["train"] and active["rollout"])
    assert overlap and not any(active.values())
    save(f"real-q{q}-k{k}-c{cadence}-lag{lag}", dict(config=asdict(pipe._config),
         parameter_count=vector(train.model).numel(), report=asdict(report), real_events=audit.events,
         updates=train.records, transfers=sync.transfers, initial_fingerprint=train.snapshots[0],
         final_fingerprint=fingerprint(train.model), tensor_device=str(next(train.model.parameters()).device)))


def test_bridge_initialize_aligns_real_weights():
    rollout, train, sync, _ = engines()
    rollout.model = TinyPolicy(seed=999)
    assert fingerprint(rollout.model) != fingerprint(train.model)
    bridge = DualEngineBridge(train_engine=train, rollout_engine=rollout, weight_sync=sync,
                              train_devices=("cpu:train",), rollout_devices=("cpu:rollout",))
    bridge.initialize()
    try:
        assert fingerprint(rollout.model) == fingerprint(train.model), "initialize advertised alignment but copied no tensors"
    finally:
        bridge.close()


def test_partial_sync_failure_blocks_poisoned_rollout():
    coordinator = PolicyPipelineCoordinator()
    rollout, train, _, audit = engines(coordinator)
    sync = TorchSync(train, rollout, audit, coordinator)
    bridge = DualEngineBridge(train_engine=train, rollout_engine=rollout, weight_sync=sync,
                              coordinator=coordinator, train_devices=("cpu:train",),
                              rollout_devices=("cpu:rollout",))
    bridge.initialize()
    sync.fail_partial = True
    batch = ready_batch(rollout)
    assert bridge.train(batch).stepped
    with pytest.raises(RuntimeError, match="partial tensor transfer"):
        bridge.sync()
    save("partial-transfer", dict(version=coordinator.rollout_policy_version,
         actual=fingerprint(rollout.model), old=train.snapshots[0], new=train.snapshots[1]))
    try:
        assert coordinator.rollout_policy_version == 0
        assert fingerprint(rollout.model) not in (train.snapshots[0], train.snapshots[1])
        with pytest.raises((BridgeStateError, PipelineClosed)):
            bridge.rollout_session(prompts(2)[1], timeout_s=0.1)
    finally:
        bridge.close()


def test_batch_owns_routed_tensor_snapshot():
    rollout, _, _, _ = engines()
    prompt = prompts(1)[0]
    result = rollout.generate(prompt, 0)
    routes = torch.zeros((len(prompt.input_tokens) + len(result.sequences[0].resp_tokens) - 1, 1, 1), dtype=torch.int16)
    result.sequences[0].routed_experts = routes
    batch = build_batch_envelope(run_id="ownership", batch_id="b0", epoch=0, policy_version=0,
                                prompt_items=[prompt], rollout_results=[result],
                                rewards=rollout.records["0"]["rewards"], eos_token_id=2)
    routes.fill_(7)
    assert not bool(batch.sequences[0].routed_experts.any()), "producer mutation changed an already materialized batch"


@pytest.mark.parametrize("payload", ["graph", "non_cpu"])
def test_batch_rejects_non_cpu_or_graph_features(payload):
    rollout, _, _, _ = engines()
    feature = torch.ones(2, requires_grad=True) * 2 if payload == "graph" else torch.empty(2, device="meta")
    prompt = AsyncPrompt(prompt_id="0", input_tokens=(1, 3), record={"features": {"pixels": feature}})
    result = rollout.generate(prompt, 0)
    with pytest.raises(BatchContractError):
        build_batch_envelope(run_id="payload", batch_id="b0", epoch=0, policy_version=0,
                             prompt_items=[prompt], rollout_results=[result],
                             rewards=rollout.records["0"]["rewards"], eos_token_id=2)


def test_stale_loss_matches_behavior_policy_clipped_reference():
    # This explicitly probes behavior-policy GRPO semantics. The recovered
    # implementation plan keeps the existing loss, so this is a diagnostic
    # comparison, not a regression gate or proof that the existing loss was
    # unintended for its original synchronous use.
    rollout, train, _, _ = engines()
    batch = ready_batch(rollout)
    pack = _pack_train_data(make_train_pack(list(batch.sequences)))
    with torch.no_grad():
        train.model.head.weight.add_(torch.arange(train.model.head.weight.numel()).reshape_as(train.model.head.weight) * 0.02)
    logprobs, mask, advantages, old = reference_values(train.model, batch.sequences)
    loss, stats = grpo_loss_fn(pack, logprobs)
    ratio = (logprobs - old).exp()
    reference = -(torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages) * mask).sum() / mask.sum()
    actual_grad = torch.autograd.grad(loss, tuple(train.model.parameters()), retain_graph=True)
    reference_grad = torch.autograd.grad(reference, tuple(train.model.parameters()))
    error = max(float((actual - expected).abs().max()) for actual, expected in zip(actual_grad, reference_grad, strict=True))
    save("off-policy-semantics", dict(actual_loss=float(loss.detach()), reference_loss=float(reference.detach()),
         actual_ratio_mean=float(stats["ratio_mean"]), behavior_ratio_mean=float(ratio[mask].mean().detach()),
         max_gradient_error=error))
    assert error < 1e-6, f"behavior-policy GRPO gradient differs by {error}"
