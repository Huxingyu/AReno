# Async policy CPU validation

Opt-in audit of the first two validation layers. These files deliberately use
`*_cases.py` names: known broken contracts are run explicitly, without adding
expected failures or silently changing the normal CPU suite.

The runner uses only dependencies already declared by AReno plus pytest.
It disables CUDA visibility and caps Torch/OpenMP CPU threads. Each test
function (including its parameterized cases) runs in an isolated subprocess;
pytest dumps thread stacks after 15 seconds, and the runner kills only that
subprocess group after 45 seconds. Any failing invariant produces exit code 1.

```bash
python3 tests/async_policy_validation/run.py --output-dir runs/async-policy-cpu/recheck
python3 tests/async_policy_validation/run.py --layer protocol --output-dir runs/async-policy-cpu/layer1-final
python3 tests/async_policy_validation/run.py --layer torch --output-dir runs/async-policy-cpu/layer2
python3 tests/async_policy_validation/run.py --case partial_sync --output-dir runs/async-policy-cpu/partial
```

`known_cases.py` preserves the four earlier review reproductions. The protocol
cases add data-source and shutdown faults, capacity/lag invariants, and repeated
lifecycle cleanup. The matrix has 54 Q/K/C/lag configurations × 4 fixed seeds;
each run contains 16 complete prompt groups. OS scheduling can change the count
of stale batches, so the checks constrain invariants rather than exact outcomes.

The real CPU model has 425 parameters: a token embedding, tanh, and a linear
next-token head. It has no attention, KV cache, CUDA or NCCL. Two separate model
instances perform genuine multinomial sampling, autograd, SGD with momentum,
and state-dict copying. Model seed is 123; sampling seed is 41 plus a per-prompt
offset. All tensors are CPU float32. No model or dataset download is required.

The adapter reuses production `make_train_pack`, `_pack_train_data`,
`packed_next_token_logprobs`, and `grpo_loss_fn`. An independent row-wise
PyTorch calculation checks masks, advantages, logprobs, gradients and optimizer
updates. Within-step tolerance is atol=5e-7 with rtol=1e-6 for logprobs/loss and
rtol=1e-5 for gradients/parameters. Integer and mask comparisons are exact.

The callback in the experimental pipeline only accepts a prompt and a sample
index. The test adapter keeps generated results by prompt ID so the callback
can return a reward calculated from the actual completion. This is a test
adapter, not a production reward API change. The pipeline still calls the
engines directly; `DualEngineBridge` is audited separately.

The behavior-policy clipped-GRPO case intentionally uses a different oracle
from the per-step compatibility checks. Its failure records a numerical
difference: the existing loss uses a self-detached denominator. The design
recovered from `docs/ospp-async-policy@1481411`, section 12 of
`docs/projects/ospp-async-policy/IMPLEMENTATION_PLAN.md`, explicitly acknowledges
this limitation and keeps the existing loss for the first release. This case
is a diagnostic comparison, not a regression acceptance gate or a newly
introduced bug. Raw assertions and results remain unchanged; the full audit
therefore returns nonzero for this diagnostic even if all contract cases pass.
Quality validation remains part of TODO T12 / T14.

## Independent synchronous baseline

Run from the repository root. The script explicitly puts the selected source
worktree first on the import path, while using the same test model and seeds.

```bash
git worktree add --detach ../areno-cpu-baseline 60e2fa7
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 tests/async_policy_validation/reference.py --repo-root ../areno-cpu-baseline --mode sync --output runs/async-policy-cpu/reference-baseline.json
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 tests/async_policy_validation/reference.py --repo-root . --mode async --output runs/async-policy-cpu/reference-candidate.json --compare runs/async-policy-cpu/reference-baseline.json
git worktree remove ../areno-cpu-baseline
```

This compares four prompts / eleven completions through materialization and
one optimizer step. The baseline calls the real synchronous
`PolicyOnlyTrainer._materialize_train_batch`; the candidate builds an async
envelope. Both execute the same real CPU tensor backend. It does not demand
that unconstrained asynchronous training follow the synchronous loss curve.

The measured report for this session is in `runs/async-policy-cpu/report.html`,
with a Markdown counterpart, a full flow diagram, the real stage timeline,
JUnit XML, JSON records and per-case logs. `render_report.py` regenerates the
report from the saved `layer1-final`, `layer2` and `reference-comparison.json`
artifacts; its narrative is specific to this audit campaign.
