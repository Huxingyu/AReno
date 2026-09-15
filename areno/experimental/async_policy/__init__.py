"""Experimental completion pipeline; imports do not load Torch or GPU kernels.

Adapters implement the protocols in ``contracts``. The production GPU adapter
and public Trainer/CLI integration are deliberately a separate validation stage.
"""

from .bridge import DualEngineBridge
from .contracts import (
    AsyncPolicyConfig,
    AsyncPrompt,
    BatchContractError,
    BatchEnvelope,
    BatchMetric,
    BridgeStateError,
    DeviceMode,
    Lifecycle,
    PipelineClosed,
    PipelineReport,
    RolloutEngine,
    RolloutSession,
    ShutdownTimeout,
    SyncMetric,
    SyncPlan,
    TrainEngine,
    TrainResult,
    WeightSync,
)
from .coordination import (
    BoundedReadyQueue,
    InflightClosed,
    InflightLimiter,
    PolicyPipelineCoordinator,
    QueueAborted,
    QueueEmpty,
    QueueFinished,
)
from .data import build_batch_envelope
from .pipeline import AsyncPolicyPipeline

__all__ = [
    "AsyncPolicyConfig", "AsyncPolicyPipeline", "AsyncPrompt", "BatchContractError",
    "BatchEnvelope", "BatchMetric", "BoundedReadyQueue", "BridgeStateError", "DeviceMode",
    "DualEngineBridge", "InflightClosed", "InflightLimiter", "Lifecycle", "PipelineClosed",
    "PipelineReport", "PolicyPipelineCoordinator", "QueueAborted", "QueueEmpty", "QueueFinished",
    "RolloutEngine", "RolloutSession", "ShutdownTimeout", "SyncMetric", "SyncPlan",
    "TrainEngine", "TrainResult", "WeightSync", "build_batch_envelope",
]
