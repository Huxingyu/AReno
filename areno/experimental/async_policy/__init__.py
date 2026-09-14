"""Experimental async policy trainer primitives.

This package holds the protocol foundation for the async GRPO trainer:
the immutable ready-batch contract, the bounded ready queue, inflight
permits, and the policy pipeline coordinator. None of these modules import a
GPU backend, so they stay testable on CPU.
"""

from areno.experimental.async_policy.batch import (
    BatchContractError,
    BatchEnvelope,
    GroupStat,
    build_batch_envelope,
)
from areno.experimental.async_policy.coordinator import (
    PipelineClosed,
    PolicyPipelineCoordinator,
    PolicySyncPlan,
)
from areno.experimental.async_policy.queue import (
    BoundedReadyQueue,
    InflightClosed,
    InflightLimitExceeded,
    InflightLimiter,
    QueueAborted,
    QueueClosed,
    QueueEmpty,
    QueueFull,
    QueuedBatch,
)

__all__ = [
    "BatchContractError",
    "BatchEnvelope",
    "BoundedReadyQueue",
    "GroupStat",
    "InflightClosed",
    "InflightLimitExceeded",
    "InflightLimiter",
    "PipelineClosed",
    "PolicyPipelineCoordinator",
    "PolicySyncPlan",
    "QueueAborted",
    "QueueClosed",
    "QueueEmpty",
    "QueueFull",
    "QueuedBatch",
    "build_batch_envelope",
]
