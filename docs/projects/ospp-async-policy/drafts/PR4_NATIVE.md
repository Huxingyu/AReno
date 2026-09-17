# feat(experimental): add native async training and resumable validation

Connect the experimental policy pipeline to native train and rollout workers
on separate GPUs. The adapter uses public lifecycle interfaces, preserves the
LoRA seed, bounds RPC waits, and requests shutdown on both partitions before
waiting for either one to exit.

The runnable example supports synchronous controls, policy lag, GRPO/GSPO,
an optional off-policy GRPO loss, and checkpoint continuation. Each update
records policy versions, prompt IDs, rewards, response lengths and token-limit
hits. The supported topology is text-only, TP=1/DP=1, with one GPU per role.

Experiment tools run individual cases with durable attempt records, committed
source provenance, whole-controller deadlines and a shared spending ledger.
Training checkpoints are persisted before independent evaluation starts;
evaluation retries reuse completed work and never repeat successful training.
Hash-verified caches cover model/tokenizer weights, data, sampling, evaluator
implementation and runtime. Summaries retain failed attempts and missing
controls, and mark an incomplete matrix explicitly. Prefill/decode timings and
generated tokens are measured separately from tokens used for updates.

## Validation

On `b5d01f4`, with CUDA hidden, using `python -m pytest tests/ -k cpu` and
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`:

- **950 passed, 12 failed, 10 skipped, 22 deselected**. The comparison baseline
  `a35a580` has 889 passed and exactly the same 12 failing node IDs. All original
  selected tests remain; 61 selected tests were added. The full suite is not green.
- The shared failures concern agentic metadata, AV event aggregation, Bailing
  cache slots, Gemma video FPS metadata and MoE sequence parallelism.
- Related campaign/benchmark checks: **34 passed**. Changed Python files pass
  Ruff, and the review diff passes whitespace checks. Sphinx HTML builds with
  warnings treated as errors.

A real Modal gate on `fa4b2f7` completed a four-update synchronous run and
128-question evaluation at both 64 and 256 tokens. The first evaluation timed
out; its retry reused three of four cached evaluations, finished the remaining
one, and verified the checkpoint was unchanged. Re-entering the completed case
created no new attempts or spending entries. Failed-attempt history is retained
by `9dd3e60`. All gate apps were confirmed stopped. This is a functional gate,
not a completed quality matrix or GPU validation of later timing instrumentation.

Full-parameter checkpoint restoration has an exact boundary audit, but the
default native path still fails strict post-resume bitwise equality. A separate
deterministic-kernel branch is under validation. Expanded quality/performance
matrices, multi-group generation, and larger topologies remain incomplete.

## Dependencies and review range

Depends on #563 (sampling seeds), #564 (worker lifecycle/checkpoints), and #565
(pipeline core). Head: `review/async-native` at `b5d01f4`. Its own changes can be
reviewed with `git diff 79a8798..b5d01f4`; the GitHub diff includes prerequisite
merges until those PRs land. Project-process documents and run artifacts are
excluded from the branch.
