# feat(engine): expose bounded worker lifecycle

An asynchronous caller needs to observe RPC readiness without consuming the
request, cancel partition startup, and verify worker exits within a shutdown
budget. Add public cluster interfaces for these operations so callers can use
the engine without accessing pending-call or result-pump internals.

## Changes

- Add `ClusterCallHandle.done()` and `wait(timeout)`, preserving the existing
  `result(timeout)` behavior.
- Add `request_shutdown()` and `close(timeout_s=...)`, with observable
  `WorkerExit` records, pending-caller wakeup, and terminate/kill escalation.
  A finite close shares its budget across all ranks and the result pump.
- Add a shared partition readiness deadline and stop event. Startup failure
  cleanup has a **separate shared five-second budget**; it can extend the time
  before the startup exception is returned.
- Add single-rank training-state save/load operations for optimizer, RNG, and
  step state, with checkpoint identity validation.

The default close retains the existing per-worker grace/terminate path.
Rejecting submissions after closure and waking pending callers are intentional
failure-handling changes. A failed bounded close retains exit evidence and can
be retried. Training-state persistence currently requires TP=1 and DP=1.

## Validation

On `ac0cd7c`, with CUDA hidden: **33 passed**. Tests cover lifecycle races and
timeouts, partition cleanup, policy tensor synchronization, real CPU optimizer
continuation, stable backend behavior, and import boundaries.

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=. python -m pytest -q \
  tests/test_protocol_cpu.py \
  tests/test_parallel_partition_cpu.py \
  tests/test_policy_tensor_sync_cpu.py \
  tests/test_training_state_cpu.py \
  tests/test_dual_engine_backend_cpu.py \
  tests/test_import_boundaries_cpu.py
```

GPU worker faults, NCCL timeout cleanup, and GPU checkpoint continuation must
be rerun on this implementation before GPU acceptance is claimed.

## Dependencies

None. Base: `48d07c5`. Local branch: `review/async-engine`. The native integration
uses these interfaces in PR 4.
