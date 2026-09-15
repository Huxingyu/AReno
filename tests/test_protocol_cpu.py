from __future__ import annotations

import importlib.util
import queue
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_protocol_module():
    """Load protocol.py without importing areno.engine package side effects."""

    path = Path(__file__).resolve().parents[1] / "areno" / "engine" / "protocol.py"
    spec = importlib.util.spec_from_file_location("_areno_protocol_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_protocol_module()
TPCluster = protocol.TPCluster
Op = protocol.Op
WorkerResult = protocol.WorkerResult


class FakeQueue:
    """Small queue double that records close/join_thread calls."""

    def __init__(self):
        self.closed = False
        self.joined = False
        self.items = []

    def put(self, item):
        self.items.append(item)

    def close(self):
        self.closed = True

    def join_thread(self):
        self.joined = True

    def cancel_join_thread(self):
        self.cancelled = True


class FakeProcess:
    """Small process double for TPCluster.close resource cleanup tests."""

    def __init__(self, alive: bool):
        self._alive = alive
        self.join_calls = []
        self.terminated = False
        self.pid = 123
        self.exitcode = None if alive else 0

    def join(self, timeout=None):
        self.join_calls.append(timeout)

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False
        self.exitcode = -15

    def kill(self):
        self._alive = False
        self.exitcode = -9

    def close(self):
        assert not self._alive


def fake_cluster(processes=()):
    cluster = object.__new__(TPCluster)
    cluster.config = SimpleNamespace(tp_size=1, dp_size=len(processes) or 1)
    cluster.started = True
    cluster.cmd_queues = [FakeQueue() for _ in processes]
    cluster.result_queue = FakeQueue()
    cluster.processes = list(processes)
    cluster._send_lock = threading.Lock()
    cluster._pending_lock = threading.Lock()
    cluster._pending_calls = {}
    cluster._pump_stop = threading.Event()
    cluster._pump_thread = None
    cluster._closed = False
    cluster._cleanup_complete = False
    cluster.worker_exits = []
    return cluster


class TPClusterResourceTest(unittest.TestCase):
    """Protocol resource tests avoid spawning real multiprocessing workers."""

    def test_close_closes_command_and_result_queues(self):
        """TPCluster.close should release queue semaphores after worker shutdown."""
        cluster = fake_cluster([FakeProcess(alive=False), FakeProcess(alive=True)])

        cluster.close()

        self.assertFalse(cluster.started)
        self.assertFalse(cluster.processes[1].is_alive())
        self.assertTrue(cluster.processes[1].terminated)
        self.assertEqual(cluster.processes[0].join_calls, [5, 0])
        self.assertEqual(cluster.processes[1].join_calls, [5, 0])
        for channel in [*cluster.cmd_queues, cluster.result_queue]:
            self.assertTrue(channel.closed)
            self.assertTrue(channel.joined)

    def test_async_call_can_wait_for_user_visible_rollout_ranks_only(self):
        """Async rollout futures should not wait for TP sibling acks before returning."""

        cluster = fake_cluster()
        cluster.config = SimpleNamespace(tp_size=2, dp_size=2)
        cluster.started = True
        cluster.cmd_queues = [FakeQueue() for _ in range(4)]
        cluster._pending_lock = threading.Lock()
        cluster._send_lock = threading.Lock()
        cluster._pending_calls = {}

        pending = cluster._submit_call(Op.INFER_ROLLOUT, request_id=7, result_ranks={0, 2})

        cluster._apply_result(7, 1, WorkerResult(ok=True, payload="tp-sibling"), pending)
        self.assertFalse(pending.event.is_set())

        cluster._apply_result(7, 0, WorkerResult(ok=True, payload="dp0"), pending)
        self.assertFalse(pending.event.is_set())

        cluster._apply_result(7, 2, WorkerResult(ok=True, payload="dp1"), pending)
        self.assertTrue(pending.event.is_set())
        self.assertEqual(pending.results[0], "dp0")
        self.assertEqual(pending.results[2], "dp1")


def test_handle_poll_timeout_does_not_cancel_and_close_wakes_waiter():
    cluster = fake_cluster([FakeProcess(alive=False)])
    pending = cluster._submit_call(Op.TRAIN, request_id=1)
    handle = protocol.ClusterCallHandle(cluster, 1, pending)
    assert not handle.done() and not handle.wait(0.001)
    assert cluster._pending_calls[1] is pending
    cluster.close(timeout_s=0.5)
    assert handle.wait(0) and handle.done()
    with pytest.raises(RuntimeError, match="cluster closed"):
        handle.result(timeout=0)
    with pytest.raises(RuntimeError, match="cluster closed"):
        cluster._submit_call(Op.TRAIN, request_id=2)
    assert cluster.close(timeout_s=0) == cluster.worker_exits


def test_bounded_close_escalates_and_keeps_exit_information():
    class IgnoreTerminate(FakeProcess):
        def terminate(self):
            self.terminated = True

    process = IgnoreTerminate(alive=True)
    cluster = fake_cluster([process])
    channels = [*cluster.cmd_queues, cluster.result_queue]
    exits = cluster.close(timeout_s=0.1)
    assert exits == [protocol.WorkerExit(0, 123, -9, False)]
    assert process.terminated and not cluster.processes
    assert all(channel.cancelled and channel.closed and not channel.joined for channel in channels)


def test_bounded_close_timeout_retains_survivors_for_retry():
    class SurvivesKill(FakeProcess):
        def terminate(self):
            pass

        def kill(self):
            pass

    process = SurvivesKill(alive=True)
    cluster = fake_cluster([process])
    with pytest.raises(TimeoutError, match="shutdown incomplete"):
        cluster.close(timeout_s=0)
    assert cluster.worker_exits == [protocol.WorkerExit(0, 123, None, True)]
    assert cluster.processes == [process]
    process._alive, process.exitcode = False, -9
    assert cluster.close(timeout_s=0.5)[0].alive is False


@pytest.mark.parametrize("cancelled", [False, True])
def test_partitioned_start_timeout_or_stop_aborts_both_partitions(monkeypatch, cancelled):
    train = protocol.ClusterPartition("train", 0, 1, 1, (0,))
    rollout = protocol.ClusterPartition("rollout", 1, 1, 1, (1,))
    world = protocol.DistributedWorldSpec("127.0.0.1", 0, 2, train, rollout)
    clusters = [fake_cluster([FakeProcess(alive=True)]) for _ in range(2)]
    for cluster, partition in zip(clusters, (train, rollout), strict=True):
        cluster.world_spec, cluster.partition = world, partition
        cluster.started = False
        cluster._spawn_workers = lambda: None
        cluster.result_queue = queue.Queue()
        cluster.result_queue.cancel_join_thread = lambda: None
        cluster.result_queue.close = lambda: None
    monkeypatch.setattr(protocol, "_create_rendezvous_store", lambda *args: SimpleNamespace(port=12345))
    stop = threading.Event()
    if cancelled:
        stop.set()
    start = time.monotonic()
    with pytest.raises(InterruptedError if cancelled else TimeoutError):
        protocol.start_partitioned_clusters(*clusters, world, timeout_s=0.01, stop_event=stop)
    assert time.monotonic() - start < 0.5
    assert all(not cluster.started and not cluster.processes for cluster in clusters)
    assert all(cluster.worker_exits[0].exitcode == -15 for cluster in clusters)


def test_completion_racing_close_cannot_overwrite_the_first_result():
    cluster = fake_cluster([FakeProcess(alive=False)])
    pending = cluster._submit_call(Op.TRAIN, request_id=1)
    cluster._apply_result(1, 0, WorkerResult(ok=True, payload="ready"), pending)
    cluster._finish_pending_call(1, pending, RuntimeError("late failure"))
    assert protocol.ClusterCallHandle(cluster, 1, pending).result(0) == ["ready"]
    cluster.close(timeout_s=0.5)


if __name__ == "__main__":
    unittest.main()
