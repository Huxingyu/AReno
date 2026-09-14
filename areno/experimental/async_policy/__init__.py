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
from areno.experimental.async_policy.bridge import (
    BridgeConfigError,
    BridgeStateError,
    DeviceMode,
    DualEngineBridge,
)
from areno.experimental.async_policy.config import AsyncPolicyConfig
from areno.experimental.async_policy.coordinator import (
    PipelineClosed,
    PolicyPipelineCoordinator,
    PolicySyncPlan,
)
from areno.experimental.async_policy.events import EventTimeline, TimelineEvent
from areno.experimental.async_policy.fake_backend import (
    AsyncPrompt,
    FakeRolloutEngine,
    FakeTrainEngine,
    FakeWeightSync,
)
from areno.experimental.async_policy.metrics import BatchMetric, MetricsCollector
from areno.experimental.async_policy.pipeline import AsyncPolicyPipeline, PipelineReport
from areno.experimental.async_policy.queue import (
    BoundedReadyQueue,
    InflightClosed,
    InflightLimiter,
    InflightLimitExceeded,
    QueueAborted,
    QueueClosed,
    QueuedBatch,
    QueueEmpty,
    QueueFull,
)

__all__ = [
    "AsyncPolicyConfig",
    "AsyncPolicyPipeline",
    "AsyncPrompt",
    "BatchContractError",
    "BatchEnvelope",
    "BatchMetric",
    "BoundedReadyQueue",
    "BridgeConfigError",
    "BridgeStateError",
    "DeviceMode",
    "DualEngineBridge",
    "EventTimeline",
    "FakeRolloutEngine",
    "FakeTrainEngine",
    "FakeWeightSync",
    "GroupStat",
    "InflightClosed",
    "InflightLimitExceeded",
    "InflightLimiter",
    "MetricsCollector",
    "PipelineClosed",
    "PipelineReport",
    "PolicyPipelineCoordinator",
    "PolicySyncPlan",
    "QueueAborted",
    "QueueClosed",
    "QueueEmpty",
    "QueueFull",
    "QueuedBatch",
    "TimelineEvent",
    "build_batch_envelope",
]
