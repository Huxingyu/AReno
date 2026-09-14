"""Collect the 30 migrated regression contracts in the normal CPU suite.

The alternate-loss S1 diagnostic is intentionally available only via run.py.
"""

from tests.async_policy_validation.known_cases import (  # noqa: F401
    test_begin_sync_honors_timeout,
    test_max_steps_does_not_report_success_with_live_producer,
    test_only_one_sync_lease_is_admitted,
    test_sync_mode_rejects_rollout_during_active_training,
)
from tests.async_policy_validation.protocol_cases import (  # noqa: F401
    test_batch_wait_metric_includes_empty_polls,
    test_data_source_iter_error_reaches_consumer,
    test_data_source_next_error_releases_permit,
    test_empty_data_and_repeated_lifecycle,
    test_full_queue_stop_releases_all_producers,
    test_inflight_close_and_release_rejects_waiter,
    test_queue_abort_rejects_buffered_batches,
    test_repeated_nonempty_runs_release_objects_and_threads,
    test_seeded_capacity_and_lag_matrix,
    test_stage_failure_propagates_and_releases_resources,
    test_train_failure_report_records_started_run,
)
from tests.async_policy_validation.torch_cases import (  # noqa: F401
    test_batch_owns_routed_tensor_snapshot,
    test_batch_rejects_non_cpu_or_graph_features,
    test_bridge_initialize_aligns_real_weights,
    test_partial_sync_failure_blocks_poisoned_rollout,
    test_real_cpu_pipeline_versions_gradients_and_sync,
)
