"""Keep startup/warmup work out of the reported steady-state throughput."""

from types import SimpleNamespace

from tests.async_policy_validation.benchmark_run import measure


def test_benchmark_excludes_warmup_tokens_and_updates(tmp_path):
    updates = [{"stepped": True, "start_ns": index * 10**9, "end_ns": (index + 1) * 10**9,
                "response_tokens": 10_000 if index < 5 else 100, "rewards": [1.0]}
               for index in range(55)]
    summary = {"updates": updates, "batch_metrics": [{}] * 55,
               "sync_metrics": [{"target_version": index, "transfer_s": 0.1, "wait_s": 0.2}
                                for index in range(56)]}
    monitor = SimpleNamespace(errors=[], rows=[
        {"time_ns": second * 10**9, "cpu_percent": 100.0, "summed_process_rss_bytes": 1024,
         "gpus": [{"device": device, "utilization_percent": 50.0, "memory_mib": 2048, "power_watts": 20.0}
                  for device in (0, 1)]}
        for second in (1, 10, 20, 40, 60)
    ])
    result = measure(summary, tmp_path, monitor)
    assert result["measured_updates"] == 50
    assert result["duration_s"] == 50
    assert result["updates_per_s"] == 1
    assert result["trained_response_tokens"] == 5000
    assert result["trained_response_tokens_per_s"] == 100
    assert result["monitor_samples"] == 3
