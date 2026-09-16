Experimental async policy training
==================================

AReno's experimental completion pipeline overlaps sample production with
policy training on separate GPUs. This can reduce idle time when completions
are long, each prompt has many samples, or CPU reward evaluation is slow.
Agentic environments motivate the design but are not supported by the native
adapter yet. Throughput and training quality must be evaluated separately.

The entry point is ``areno.experimental.async_policy.AsyncPolicyPipeline``.
The native adapter currently supports text completion with one training GPU
and one rollout GPU, each with TP=1 and DP=1. See
:doc:`../cookbook/async-grpo-two-gpu` for the runnable example.

Two stages and one weight owner
-------------------------------

.. code-block:: text

   Production: at most K prompt groups
   read prompt -> rollout GPU -> CPU reward + group advantages -> ready queue Q
                                                                  |
   Training:                                                     v
   training GPU <- admit batch and check policy lag <- consume whole group
        |
        +-> actual optimizer update -> synchronization due?
                                       |
                         stop new model admission; finish active operations
                                       |
                         copy weights -> acknowledge -> reopen admission

The coordinator owns both policy versions. The training version advances only
when the worker reports an actual optimizer update. Each batch captures the
rollout version used to generate it. Initial synchronization copies weights
even when both counters are zero; version equality alone does not prove that
the models agree.

One rollout session and one training operation can overlap. Synchronization
is exclusive with both. Published training batches contain independently
owned CPU data without autograd graphs. Rewards and advantages are computed
for complete prompt groups before publication.

Configuration and units
-----------------------

``AsyncPolicyConfig`` provides these bounds:

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - Field
     - Default
     - Unit and meaning
   * - ``queue_capacity`` (Q)
     - 2
     - Ready prompt-group batches waiting for training.
   * - ``max_inflight_rollouts`` (K)
     - 1
     - Prompt groups, from reading a prompt through publishing its batch.
   * - ``rollout_batch_groups``
     - 1
     - Maximum complete groups in one generation session; cannot exceed K.
   * - ``weight_sync_interval_updates`` (C)
     - 1
     - Successful optimizer updates since the last weight copy.
   * - ``max_policy_lag``
     - 1
     - Maximum accepted difference between training and batch versions, inclusive.
   * - ``max_steps``
     - ``None``
     - Successful updates in this invocation. Zero disables training.
   * - ``operation_timeout_s``
     - 60
     - Seconds available for a model operation and its admission wait.
   * - ``shutdown_timeout_s``
     - 60
     - Seconds shared by pipeline cleanup operations.
   * - ``poll_interval_s``
     - 0.05
     - Seconds between consumer queue polls.

K includes groups doing CPU scoring or waiting to publish. Increasing it does
not enable concurrent GPU generation sessions. Queue and inflight bounds limit
batch counts, not bytes; longer responses or more samples still use more memory.

With ``rollout_batch_groups > 1``, the adapter implements
``BatchedRolloutEngine.generate_batch`` and returns a dictionary keyed by each
group's input position. One session captures one policy version for all groups.
The bridge validates and copies every result before releasing its lease. Each
group is then scored independently and published as one training batch, with
the original optimizer-update boundary. Group identity is positional, so repeated
prompt IDs and texts do not merge groups.

The producer packs only permits that are immediately available. It submits a
partial batch at EOF or when no additional permit is available; it never waits
to fill a batch while holding the generation lease. The source iterator must
still be cooperative. Synchronization retains priority over the next session.
Packing may change sampling RNG consumption, staleness and the update trajectory;
compare quality as well as throughput before enabling it for a workload.

Staleness and the loss
----------------------

A batch is accepted when ``0 <= train_version - batch_version <= max_policy_lag``.
Older batches are discarded whole; a future version is an error. The lag bound
controls admission and can force an earlier weight copy. With one update per
batch, synchronization becomes due after ``min(C, max_policy_lag + 1)`` updates.
For example, C=4 with lag=1 synchronizes after two updates, so that setting
cannot measure a four-update cadence.

Lag does not select a loss or correct behavior-policy bias. The default reuses
upstream ``grpo_loss_fn`` unchanged. Its forward ratio
``exp(logp - logp.detach())`` equals one, while its policy gradient remains
nonzero. It does not use recorded rollout log probabilities as the denominator.
The example offers ``--loss grpo-offpolicy`` as a separate clipped-ratio
ablation and ``--loss gspo`` to exercise the existing GSPO loss. These options
do not establish quality or convergence guarantees.

Lag zero requires current-version batches at training admission. Speculative
rollouts may still finish after an update and be discarded. It therefore does
not guarantee the sample order, RNG consumption or optimization trajectory of
a serial synchronous loop. Use the example's ``--mode sync`` for that control.

Ending a run
------------

.. list-table::
   :header-rows: 1
   :widths: 22 25 53

   * - Trigger
     - Reported exit reason
     - Behavior
   * - Source EOF
     - ``data_exhausted``
     - Finish accepted production tasks, drain ready batches with lag checks,
       and perform a final weight synchronization.
   * - Update limit
     - ``max_steps``
     - Stop admission and discard remaining work. The rollout copy may lag the
       final training checkpoint.
   * - ``request_stop()``
     - ``stopped``
     - Wake waiters, cancel pending consumption and clean up owned resources.
   * - ``abort()``
     - ``aborted``
     - Stop without draining buffered batches.
   * - Operation or cleanup failure
     - ``failed``
     - Preserve and raise the first exception; retain secondary cleanup errors
       and report resources that have not exited.

Reward callbacks receive the same ``RewardRecord`` used by the upstream
training code. They and the source iterator must return promptly or cooperate
with cancellation. Python cannot safely kill an arbitrary callback thread.
A cleanup timeout is a failure, not successful background cleanup; callers
can retry ``close()`` after the blocked work exits.

Integration limits
------------------

The experimental bridge directly uses native engine operations. The stable
``CudaBackend`` owns a synchronous rollout-session guard and its own policy
version counters; sharing those owners with a concurrent coordinator would
require an explicit integration design. No ``async-grpo`` algorithm is
registered in the public ``Trainer`` or main training CLI yet.

The native adapter waits for active generation before copying weights.
Multi-session generation, TP/DP greater than one, agentic trajectories and
multimodal features remain future work. Existing short-run evidence does not
establish stable training quality. The behavior-ratio ablation and new
benchmark configurations require their own GPU validation.
