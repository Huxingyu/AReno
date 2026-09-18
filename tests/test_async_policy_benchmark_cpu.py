"""Keep startup/warmup work out of the reported steady-state throughput."""

import io
import tarfile
from types import SimpleNamespace

import pytest

from examples.async_policy.tools.benchmark_run import case_plan, generation_window, measure, parse_args, quality_metrics
from examples.async_policy.tools.matrix import build_jobs, summarize
from examples.async_policy.tools.modal_run import collect_artifacts


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


def test_quality_metrics_keep_accuracy_length_and_limit_hits_separate():
    rows = [{"correct": True, "response_tokens": 30}, {"correct": False, "response_tokens": 64}]
    result = quality_metrics(rows, 64)
    assert result["accuracy"] == result["token_limit_hit_rate"] == 0.5
    assert result["mean_response_tokens"] == 47
    assert quality_metrics(rows, 256)["token_limit_hits"] == 0


def test_generation_counts_exclude_warmup_and_boundary_requests():
    def request(start, end, tokens):
        return {"kind": "rollout", "ok": True, "start_ns": start * 10**9, "end_ns": end * 10**9,
                "inference": {"returned_response_tokens": tokens - 2, "response_rows": 2,
                              "stages": {
                                  "prefill": {"calls": 1, "cpu_s": 0.2, "cuda_s": 0.1,
                                              "sampled_tokens": 2, "input_tokens": 80},
                                  "decode": {"calls": 2, "cpu_s": 0.5, "cuda_s": 0.4,
                                             "sampled_tokens": tokens - 2, "input_tokens": 0}}}}

    rows = [request(0, 9, 9999), request(9, 11, 999), request(11, 14, 50),
            request(17, 20, 100), request(19, 21, 888), request(20, 22, 9999)]
    result = generation_window(rows, 10 * 10**9, 20 * 10**9)
    assert result["available"] and result["complete_requests"] == 2
    assert result["boundary_requests_excluded"] == 2
    assert result["sampled_tokens"] == 150 and result["sampled_tokens_per_s"] == 15
    assert result["returned_response_tokens"] == 146 and result["response_rows"] == 4
    assert result["stages"]["prefill"]["input_tokens"] == 160
    assert result["stages"]["decode"]["cuda_s"] == pytest.approx(0.8)


def test_legacy_worker_logs_do_not_report_zero_generation():
    assert generation_window([{"kind": "rollout", "start_ns": 1, "end_ns": 2}], 0, 3) == {"available": False}


def test_quality_matrix_keeps_training_and_evaluation_budgets_independent():
    args = parse_args(["--model-path", "checkpoint", "--output-dir", "output", "--seed", "43",
                       "--max-new-tokens", "256", "--eval-max-new-tokens", "64", "256",
                       "--cases", "sync", "lag1", "lag0", "offpolicy"])
    cases = case_plan(args)
    assert len(cases) == 4 and len({case["label"] for case in cases}) == 4
    assert {case["case"]: case["loss"] for case in cases}["offpolicy"] == "grpo-offpolicy"
    assert args.max_new_tokens == 256 and args.eval_max_new_tokens == [64, 256]


@pytest.mark.parametrize(("suite", "job_count", "run_count"), [
    ("quality", 10, 40), ("throughput", 12, 24), ("capacity", 57, 57), ("gspo", 6, 12),
])
def test_matrix_size_and_missing_evidence(tmp_path, suite, job_count, run_count):
    args = SimpleNamespace(suite=suite, seeds=None, model_path="checkpoint", output_dir=tmp_path,
                           steps=55, warmup=5, attn_backend="native")
    jobs = build_jobs(args)
    assert len(jobs) == run_count
    assert sum(len(job["cases"]) for job in jobs) == run_count
    assert len({job["name"] for job in jobs}) == run_count
    assert len({job["name"].removesuffix("-" + job["cases"][0]) for job in jobs}) == job_count
    if suite == "capacity":
        assert all(job["lag"] >= job["weight_sync_interval_updates"] for job in jobs)
    result = summarize(jobs, tmp_path)
    assert not result["complete"] and result["completed_jobs"] == 0
    assert len(result["missing_jobs"]) == run_count


def test_benchmark_rejects_missing_measured_window(tmp_path):
    with pytest.raises(ValueError, match="warmup"):
        measure({"updates": [{"stepped": True}]}, tmp_path, SimpleNamespace(), warmup=1)


@pytest.mark.parametrize("reports_only", [False, True])
def test_artifact_reports_leave_large_checkpoints_uncompressed(tmp_path, reports_only, monkeypatch):
    output = tmp_path / "output"
    (output / "checkpoint").mkdir(parents=True)
    entries = {"checkpoint/model.safetensors": b"weights", "checkpoint/training_state.pt": b"optimizer",
               "checkpoint/training_state.json": b'{"global_step":3}', "result.json": b'{"ok":true}'}
    for name, content in entries.items():
        (output / name).write_bytes(content)
    original_add = tarfile.TarFile.add

    def checked_add(archive, name, *args, **kwargs):
        if reports_only:
            assert str(name).endswith(".json"), "large checkpoints must never enter the compressor"
        return original_add(archive, name, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "add", checked_add)
    destination = tmp_path / "volume.tar.gz"
    downloaded = collect_artifacts(output, destination, reports_only=reports_only)
    expected = {name for name in entries if name.endswith(".json")} if reports_only else set(entries)
    with tarfile.open(destination) as archive:
        assert set(archive.getnames()) == expected
    assert all((output / name).read_bytes() == content for name, content in entries.items())
    with tarfile.open(fileobj=io.BytesIO(downloaded)) as archive:
        assert set(archive.getnames()) == expected
