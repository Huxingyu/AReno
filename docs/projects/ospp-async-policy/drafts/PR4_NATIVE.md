# feat(experimental): add native async example

Connect the experimental pipeline to AReno's native train and rollout workers
using the public lifecycle interfaces. The adapter constructs engines lazily,
preserves the explicit LoRA seed, bounds RPC waits, and requests shutdown on both
partitions before waiting for either partition to exit.

Add a two-GPU training example with native attention by default, configurable
sampling and optimization settings, explicit loss selection, and checkpoint
continuation. Record policy versions, participating prompt IDs, response lengths,
token-limit hits, and rewards for subsequent quality analysis. Token-limit hits
are a truncation proxy, not an engine stop reason.

Validation helpers live under `examples/async_policy/tools/`; numerical oracles
and SDK regression fixtures remain under `tests/`. The tools include a fixed
128-question holdout, paired quality/throughput/capacity/GSPO matrices, full
parameter continuation, and bounded Modal jobs that fetch a committed branch.
The cookbook documents the supported text-only TP=1/DP=1 topology and limitations.

## Validation

On `2b0575a` with PRs 1–3 integrated:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=. python -m pytest tests/ -k cpu
make -C docs html
```

- Full CPU result: **921 passed, 12 failed, 10 skipped, 22 deselected**.
  The comparison baseline `a35a580` has **889 passed and the same 12 failing
  node IDs**. The full suite is not green.
- All 911 original selected CPU tests remain; 32 tests were added. All 149
  tests in changed test modules pass, including unittest class methods.
- Baseline and candidate Sphinx HTML builds complete without warnings.
- Helper help/preview commands and all 17 Modal resource previews succeed.
  Resource previews do not execute their GPU workloads.

The shared baseline failures concern agentic metadata, AV event aggregation,
Bailing cache slots, Gemma video FPS metadata, and MoE sequence parallelism.
The candidate and baseline use the same local Python/Torch environment.

GPU example/fault/regression/continuation checks and the expanded experiment
matrices are pending. Historical results on an earlier implementation are not
acceptance evidence for this change. Quality stability, generation batching,
double-buffered weights, agentic trajectories, and larger TP/DP topologies are
not established by these CPU results.

## Dependencies and review range

Depends on PRs 1, 2, and 3, including seed forwarding for the synchronous
comparison. Local branch: `review/async-native`; integration commit: `2b0575a`.
Review this PR's own changes with `git diff 79a8798..2b0575a`. The branch currently
contains dependency merges; update the comparison base after those PRs land.
This description is unpublished.
