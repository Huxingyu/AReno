Async GRPO on two GPUs
======================

This experimental example runs one training worker and one rollout worker on
separate GPUs. It provides a serial control with the same model and resource
allocation. Read :doc:`../concepts/async-policy` for lag, queue and exit semantics.

Environment
-----------

Use a Linux CUDA installation of AReno with two visible NVIDIA GPUs and enough
memory for the selected model, responses and sample count. Start with the
Qwen3-0.6B text model and native LoRA. Installation instructions are in
:doc:`../getting-started/installation`. The example defaults to native attention;
``--attn-backend flash`` additionally requires a working FlashAttention install.

Commands below run from the repository root. Repository model references are
resolved through ModelScope; ``--model`` also accepts a local checkpoint directory.

.. code-block:: bash

   python examples/async_policy/train.py \
     --model Qwen/Qwen3-0.6B --steps 3 --save-training-state \
     --output-dir outputs/async-fresh

The example uses visible GPU indices 0 and 1, TP=1 and DP=1 per role. It loads
the shipped 64-question arithmetic training set, scores the final numeric answer,
and repeats prompts until it reaches the requested number of optimizer updates.
This small dataset exercises the pipeline; it is not a convergence benchmark.

Outputs and success checks
--------------------------

Each invocation needs a fresh output directory:

* ``result.json`` records settings, source commit, pipeline metrics, native
  training/synchronization statistics and worker exit information. ``ok`` is
  true only after the requested updates and worker cleanup complete.
* ``updates.jsonl`` records update versions, prompt IDs, response lengths and
  rewards. Actual optimizer updates, including zero-gradient updates, count
  toward ``--steps``.
* ``samples.jsonl`` records completions, rewards, response lengths and token-limit
  hits for scored samples. Some scored samples may later be discarded as stale.
* ``checkpoint-N`` contains the first and final updated policies. With
  ``--save-training-state`` it also contains optimizer state, training RNG state
  and a manifest needed for continuation.

Inspect ``worker_exits`` for two exited workers and the pipeline's stale-drop
count. A token-limit hit is a truncation proxy: it does not identify the actual
generation stop reason. An interrupted run preserves diagnostic output and does
not report that the full requested update count succeeded.

Synchronous control and continuation
------------------------------------

.. code-block:: bash

   python examples/async_policy/train.py \
     --model Qwen/Qwen3-0.6B --mode sync --seed 41 --steps 3 \
     --output-dir outputs/sync-control

   python examples/async_policy/train.py \
     --model Qwen/Qwen3-0.6B --steps 2 --save-training-state \
     --resume-from outputs/async-fresh/checkpoint-3 \
     --output-dir outputs/async-resumed

The resumed invocation performs two additional updates, ending at version 5.
It restores policy weights, optimizer state, training RNG and the update count.
The prompt iterator starts again from the beginning, and inflight work is not
restored. Keep optimizer settings compatible with the saved manifest. The
adapter metadata supplies the saved LoRA rank and alpha.

Example arguments
-----------------

Run ``python examples/async_policy/train.py --help`` for the command-line syntax.

.. list-table::
   :header-rows: 1
   :widths: 37 23 40

   * - Argument
     - Default
     - Meaning
   * - ``--model``
     - ``Qwen/Qwen3-0.6B``
     - ModelScope reference or local checkpoint.
   * - ``--data-path``
     - Shipped ``train.jsonl``
     - JSONL rows with ``prompt`` and ``answer``; prompts must fit 128 tokens.
   * - ``--output-dir``
     - Required
     - Fresh directory for artifacts.
   * - ``--mode``
     - ``async``
     - ``async`` or serial ``sync`` control.
   * - ``--steps``
     - 55
     - Additional successful optimizer updates.
   * - ``--seed``
     - 41
     - Sampling and initial LoRA seed.
   * - ``--lag``
     - 1
     - Inclusive maximum batch age, in updates.
   * - ``--attn-backend``
     - ``native``
     - ``native`` or ``flash`` attention.
   * - ``--lora-rank`` / ``--lora-alpha``
     - 8 / 16
     - Fresh adapter rank and scaling alpha.
   * - ``--learning-rate``
     - ``1e-5``
     - Optimizer learning rate.
   * - ``--n-samples``
     - 4
     - Completions per prompt group.
   * - ``--max-new-tokens``
     - 64
     - Maximum generated tokens per completion.
   * - ``--loss``
     - ``grpo``
     - ``grpo``, experimental ``grpo-offpolicy``, or upstream ``gspo``.
   * - ``--queue-capacity``
     - 2
     - Ready prompt-group batches (Q).
   * - ``--max-inflight-rollouts``
     - 1
     - Prompt groups (K), including CPU scoring and publication.
   * - ``--rollout-batch-groups``
     - 1
     - Groups per native generation session; at most K, async mode only.
   * - ``--weight-sync-interval-updates``
     - 1
     - Maximum updates between weight copies (C); lag can force earlier copies.
   * - ``--save-training-state``
     - Off
     - Include optimizer and training RNG state in checkpoints.
   * - ``--resume-from``
     - Unset
     - Continue from a training-state checkpoint; also enables state saving.

Quality and throughput experiments
----------------------------------

The benchmark tools accept a resolved local model directory. To resolve the
example model, use ModelScope and substitute the printed directory for
``MODEL_DIRECTORY`` below:

.. code-block:: bash

   python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen3-0.6B'))"

   python examples/async_policy/tools/matrix.py \
     --suite quality --model-path MODEL_DIRECTORY --output-dir outputs/quality

The default action writes ``matrix.json`` and starts no GPU work. Review the
matrix before adding ``--execute``. The quality suite plans five seeds, training
limits of 64 and 256 tokens, and four cases: sync, lag=1, lag=0 and lag=1 with the
behavior-ratio loss. Each checkpoint is evaluated at both 64 and 256 tokens on
the fixed 128-question holdout. The original 32 questions remain its first subset.

``--suite throughput`` covers 256/512-token outputs and 4/8 samples with three
seeds. ``--suite capacity`` covers Q in 1/2/4, K in 1/2 and C in 1/2/4, with
three seeds and lag=4 so the cadence settings remain distinct. ``--suite gspo``
plans a separate GSPO comparison. These plans do not imply measured improvements.

``--suite batching`` plans three seeds with eight samples and 256-token outputs.
It holds Q=2, K=2, C=1 and lag=1 fixed, comparing one versus two groups per
async generation session, with a serial control for each seed. Its
``batching_comparisons`` match the same async settings and include both raw
metric sets. Each group remains a separate optimizer update. In the training
example, ``--max-inflight-rollouts 2 --rollout-batch-groups 2`` opts into this
path and permits up to ``2 * n_samples`` active rollout rows. The default remains
one group per session.

After execution, ``benchmark-summary.json`` reports completed and missing jobs,
per-seed accuracy, response lengths, token-limit hit rates, and comparisons
paired with the synchronous run of the same seed. An incomplete aggregation
exits nonzero. Per-case ``benchmark.json`` files retain reward curves and runtime
metrics; raw evaluation answers remain in ``evaluation-N/result.json``.

Throughput uses successful training updates and trained response tokens after
warmup. It excludes startup and checkpoint reloading. Queue/synchronization waits
can overlap model work and must not be added as if they were disjoint wall time.
Compare quality and truncation proxies separately; faster updates alone do not
establish stable learning.
