"""CPU contracts at the new native-worker adapter boundary."""

from __future__ import annotations

import multiprocessing
import signal
import threading
import time
from types import SimpleNamespace

import pytest

from areno.api.config import CudaConfig
from areno.api.models import SamplingParams, TrainSequence
from areno.experimental.async_policy.contracts import BatchEnvelope, SyncPlan
from areno.experimental.async_policy.coordination import Deadline
from areno.experimental.async_policy.native import NativeCudaEnginePair, _wait_handles


def make_pair(tmp_path, **kwargs):
    return NativeCudaEnginePair(model_path=str(tmp_path), config=CudaConfig(
        devices=[0], rollout_devices=[1], max_running_prompts=4,
    ), sampling_params=SamplingParams(max_prompt_len=128, max_new_tokens=64),
        tokenizer=None, **kwargs)


class Handle:
    def __init__(self, result=None, error=None, ready=True):
        self.value = result
        self.error = error
        self._pending = SimpleNamespace(event=threading.Event())
        if ready:
            self._pending.event.set()

    def result(self, timeout=None):
        assert self._pending.event.is_set()
        if self.error:
            raise self.error
        return self.value


def test_native_pair_observes_failed_receiver_while_publisher_pending():
    first = RuntimeError("receiver failed")
    with pytest.raises(RuntimeError) as caught:
        _wait_handles([Handle(ready=False), Handle(error=first)], Deadline(0.1))
    assert caught.value is first


def test_native_pair_collectives_have_one_deadline():
    with pytest.raises(TimeoutError):
        _wait_handles([Handle(ready=False), Handle(ready=False)], Deadline(0.01))


def test_native_pair_keeps_rank_order_for_independent_completion():
    assert _wait_handles([Handle(["train"]), Handle(["rollout"])], Deadline(1)) == [["train"], ["rollout"]]


def test_native_pair_microbatches_produce_one_real_step(tmp_path):
    pair = make_pair(tmp_path, mini_bs=2)
    rows = tuple(TrainSequence(tokens=[1, 2, 3], prompt_len=1, scalar_advantage=1.0,
                               logprobs=[0.0, -1.0, -1.0]) for _ in range(3))
    batch = BatchEnvelope("run", "batch", 0, 7, ("prompt",), (0, 3), rows)
    received = []

    def call(role, op, payload, deadline):
        received.append(payload)
        return [[{"stepped": False, "global_step": 7, "loss": 0.0},
                 {"stepped": True, "global_step": 8, "loss": 0.0}]]

    pair._call = call
    assert pair.train(batch, 7, timeout_s=1) is True
    assert len(received[0].data_packs_by_dp) == received[0].gradient_accumulation_steps == 2
    assert sum(row["stepped"] for row in pair.train_stats) == 1


@pytest.mark.parametrize("stats", [
    [{"stepped": 1, "global_step": 1}],
    [{"stepped": True, "global_step": 3}],
])
def test_native_pair_rejects_ambiguous_or_wrong_update_results(tmp_path, stats):
    pair = make_pair(tmp_path)
    batch = BatchEnvelope("run", "batch", 0, 0, ("prompt",), (0, 1),
                          (TrainSequence(tokens=[1, 2], prompt_len=1, scalar_advantage=1, logprobs=[0, -1]),))
    pair._call = lambda *args: [stats]
    with pytest.raises(RuntimeError):
        pair.train(batch, 0, timeout_s=1)


def test_native_pair_transfers_even_version_zero(tmp_path):
    from areno.engine.protocol import Op

    pair = make_pair(tmp_path)
    calls = []
    plan = ("weight metadata",)

    def submit(op, payload=None):
        calls.append((op, payload))
        return Handle([plan] if op == Op.POLICY_SYNC_PLAN else [{"version": payload.version, "bytes": 64}])

    pair._initialized = True
    pair._clusters = {role: SimpleNamespace(submit=submit) for role in ("train", "rollout")}
    pair.transfer(SyncPlan(0, 0), timeout_s=1)
    assert [op for op, _ in calls] == [Op.POLICY_SYNC_PLAN, Op.POLICY_SYNC_PLAN,
                                     Op.POLICY_SYNC_PUBLISH, Op.POLICY_SYNC_RECEIVE]
    assert pair.sync_stats[0]["target_version"] == 0


def _uncooperative_process(ready):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    threading.Event().wait(30)


def test_native_pair_reaps_uncooperative_workers_with_shared_budget(tmp_path):
    from areno.engine.protocol import TPCluster

    pair = make_pair(tmp_path)
    context = multiprocessing.get_context("spawn")
    original_processes = []
    try:
        for role in ("train", "rollout"):
            ready = context.Event()
            process = context.Process(target=_uncooperative_process, args=(ready,))
            process.start()
            original_processes.append(process)
            assert ready.wait(10)
            cluster = TPCluster(SimpleNamespace(), None)
            cluster.processes.append(process)
            pair._clusters[role] = cluster
        started = time.monotonic()
        pair.close(timeout_s=3)
        assert time.monotonic() - started < 3
        assert len(pair.worker_exits) == 2
        assert all(not row["alive"] and row["exitcode"] == -signal.SIGKILL for row in pair.worker_exits)
        pair.close(timeout_s=0)
    finally:
        for process in original_processes:
            if not process._closed:
                if process.is_alive():
                    process.kill()
                process.join(timeout=2)
                process.close()
