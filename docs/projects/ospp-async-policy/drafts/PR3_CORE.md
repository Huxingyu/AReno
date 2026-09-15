# feat(experimental): add async policy pipeline

Add an experimental completion pipeline that overlaps rollout and training on
separate resources. One coordinator owns train admission, policy versions, and
exclusive weight publication; a supervisor owns startup, cancellation, and
shutdown. Bounded queues and explicit payload ownership make this scheduling
layer testable on CPU.

The configuration defines queue capacity, in-flight prompt groups, update-based
sync cadence, and the maximum admitted policy lag. `lag=0` admits current-version
batches, but speculative generation, dropped samples, and RNG consumption can
still differ from serial synchronous execution.

The default GRPO objective is preserved. An optional experimental
`grpo_offpolicy_loss_fn` uses detached behavior log probabilities for a clipped
ratio ablation. Its mathematical tests do not establish a quality improvement.

## Backend responsibilities

| Responsibility | Stable `CudaBackend` | Experimental bridge/coordinator |
|---|---|---|
| Train admission | Reject training during an active separate rollout session | SYNC serializes model operations; ASYNC permits overlap on separate resources |
| Train version | Increment after `stepped=True` | `end_train(stepped)` commits the version once |
| Rollout version and publication | Lazy synchronization before the next rollout | Close admission, wait for active operations, transfer and acknowledge weights, then publish the rollout version |

This delivery uses an independent experimental entry point. Registering an
`async-grpo` CLI loop requires a configuration channel and an agreed SDK
lifecycle: `Trainer.init()` starts the stable backend, while its tokenizer is
unavailable before initialization. That integration decision remains open.

## Validation

On `44fe9ac`, with CUDA hidden: **90 passed**.

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=. python -m pytest -q \
  tests/test_async_policy_contracts_cpu.py \
  tests/test_async_policy_core_cpu.py \
  tests/test_async_policy_losses_cpu.py \
  tests/test_algorithms_cpu.py \
  tests/test_import_boundaries_cpu.py
make -C docs html
```

CPU coverage includes ordering, staleness admission, cancellation, payload
snapshots, synchronization exclusion, exact loss/gradient checks, masks, and
the S1 numerical oracle. The Sphinx HTML build completed without warnings.
GPU overlap, throughput, and training quality remain separate acceptance work.

## Dependencies

None. Base: `48d07c5`. Local branch: `review/async-core`. Native worker integration
and the runnable two-GPU recipe follow in PR 4. This description is unpublished.
