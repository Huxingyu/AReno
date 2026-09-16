"""Exercise interrupted campaigns, evidence integrity, reuse and spending guards."""

import contextlib
import copy
import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from examples.async_policy.tools import checkpoint_eval, matrix, modal_control
from examples.async_policy.tools.campaign_state import (
    BudgetExceeded,
    BudgetLedger,
    complete_directory,
    exclusive_lock,
    file_manifest,
    sha256_file,
    training_code_fingerprint,
    verify_manifest,
    write_json,
)
from examples.async_policy.tools.checkpoint_eval import cached_evaluate


def campaign_args(tmp_path):
    return SimpleNamespace(suite="quality", seeds=[41], model_path="checkpoint", output_dir=tmp_path,
                           steps=55, warmup=5, attn_backend="native", backend="local", max_attempts=2,
                           job_timeout_s=300, campaign_timeout_s=1800, stage="all", select=None)


def result_row(job):
    settings = matrix.expected_settings(job)
    score = {"accuracy": 0.75, "total": 128, "correct": 96,
             "token_limit_hit_rate": 0.0, "mean_response_tokens": 40.0}
    return {"case": job["cases"][0], "ok": True, "seed": job["seed"], "source_sha": "old-tools",
            "training_source_sha": "old-tools", "evaluation_source_sha": "new-tools",
            "training_code_sha256": "same-training-code", "evaluator_sha256": "same-evaluator",
            "model_sha256": "same-weights", "environment": {"torch": "same"},
            "evaluation_environment": {"torch": "same"}, "training_settings": settings,
            "checkpoint_unchanged": True, "held_out_questions": 128,
            "train_max_new_tokens": job["max_new_tokens"], "n_samples": job["n_samples"],
            "attn_backend": job["attn_backend"], "loss": settings["loss"], "lag": settings["lag"],
            "policy": {"max_steps": job["steps"], **{key: job[key] for key in (
                "queue_capacity", "max_inflight_rollouts", "weight_sync_interval_updates")}},
            "metrics": {"updates_per_s": 1.0, "warmup_updates": job["warmup"]},
            "dataset_sha256": {"train": sha256_file(matrix.ROOT / "examples/async_policy/train.jsonl"),
                               "eval": sha256_file(matrix.ROOT / "examples/async_policy/eval128.jsonl")},
            "evaluations": {str(limit): {"before": copy.deepcopy(score), "after": copy.deepcopy(score)}
                            for limit in job["eval_max_new_tokens"]}}


def publish_result(output, job, row):
    attempt = output / job["name"] / "evaluate/attempt-0001"
    write_json(attempt / "artifacts/result.json", {"ok": True, "cases": [row]})
    complete_directory(attempt, {"test_result": True})


def test_budget_keeps_interrupted_reservation_and_refuses_overspend(tmp_path):
    budget = BudgetLedger(tmp_path / "budget.json", 1.0, reserve_usd=0.1)
    budget.reserve("first", 1200, 2)
    available = budget.summary()["available_usd"]
    budget.finish("first", 1.0, confirmed_stopped=False, status="interrupted")
    assert budget.summary()["available_usd"] == available
    assert budget.summary()["unconfirmed_attempts"] == ["first"]
    with pytest.raises(BudgetExceeded):
        budget.reserve("second", 1200, 2)
    budget.finish("first", 1.0, confirmed_stopped=True, status="stopped")
    assert budget.summary()["available_usd"] > available
    assert budget.summary()["unconfirmed_attempts"] == []
    budget.reserve("second", 1200, 2)
    with pytest.raises(ValueError, match="already reserved"):
        budget.reserve("second", 1200, 2)


@pytest.mark.parametrize("limit", [float("nan"), float("inf"), 0, -1])
def test_budget_rejects_invalid_limits(tmp_path, limit):
    with pytest.raises(ValueError):
        BudgetLedger(tmp_path / "budget.json", limit)


def test_budget_cannot_be_implicitly_increased(tmp_path):
    path = tmp_path / "budget.json"
    BudgetLedger(path, 10.0)
    with pytest.raises(ValueError, match="limit cannot be changed"):
        BudgetLedger(path, 24.0)


def test_duplicate_local_controller_is_rejected(tmp_path):
    path = tmp_path / "controller.lock"
    with exclusive_lock(path), pytest.raises(RuntimeError, match="another controller"):
        with exclusive_lock(path):
            pytest.fail("a second controller acquired an owned campaign")


def test_checkpoint_manifest_detects_deleted_and_modified_weights(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"abc")
    manifest = file_manifest(tmp_path)
    weights.write_bytes(b"abd")
    with pytest.raises(ValueError, match="changed evidence"):
        verify_manifest(tmp_path, manifest)
    weights.unlink()
    with pytest.raises(ValueError, match="missing or changed"):
        verify_manifest(tmp_path, manifest)


def test_failed_evaluation_retries_without_repeating_training(tmp_path, monkeypatch):
    args = campaign_args(tmp_path)
    job = matrix.build_jobs(args)[0]
    calls = []

    def fake_child(command, timeout_s, *, log):
        output = tmp_path.__class__(command[command.index("--output-dir") + 1])
        stage = "train" if "--train-only" in command else "evaluate"
        calls.append(stage)
        if stage == "evaluate" and calls.count("evaluate") == 1:
            write_json(output / "partial.json", {"finished_limit": 64})
            raise subprocess.CalledProcessError(1, command)
        name = "training-result.json" if stage == "train" else "result.json"
        write_json(output / name, {"ok": True, "cases": [result_row(job)]})

    monkeypatch.setattr(matrix, "run_job", fake_child)
    manifest = {"training_code_sha256": "same-training-code"}
    matrix.execute(args, [job], manifest)
    assert calls == ["train", "evaluate"]
    assert (tmp_path / job["name"] / "evaluate/attempt-0001/artifacts/partial.json").exists()
    matrix.execute(args, [job], manifest)
    matrix.execute(args, [job], manifest)
    assert calls == ["train", "evaluate", "evaluate"]
    summary = matrix.summarize([job], tmp_path)
    assert summary["complete"]
    assert summary["failed_or_interrupted_attempts"] == [{
        "job": job["name"], "attempt": f"{job['name']}/evaluate/attempt-0001/attempt.json",
        "status": "failed",
    }]
    result_file = tmp_path / job["name"] / "train/attempt-0001/artifacts/training-result.json"
    result_file.unlink()
    with pytest.raises(ValueError, match="missing or changed evidence"):
        matrix.execute(args, [job], manifest)
    assert calls == ["train", "evaluate", "evaluate"]


def test_campaign_deadline_does_not_dispatch_another_case(tmp_path, monkeypatch):
    args = campaign_args(tmp_path)
    job = matrix.build_jobs(args)[0]
    monkeypatch.setattr(matrix, "run_job", lambda *a, **k: pytest.fail("expired campaign dispatched work"))
    result = matrix.run_stage(args, job, "train", {"id": 1}, deadline=time.time() - 1)
    assert result is None
    assert not list(tmp_path.rglob("attempt.json"))


def test_matrix_preview_rejects_changed_settings_in_existing_campaign(tmp_path, monkeypatch):
    command = ["matrix.py", "--suite", "quality", "--model-path", "checkpoint",
               "--output-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", command)
    assert matrix.main() == 0
    monkeypatch.setattr(sys, "argv", [*command, "--learning-rate", "0.02"])
    with pytest.raises(ValueError, match="campaign settings/source changed"):
        matrix.main()


def test_cache_reuses_results_and_keeps_failed_attempts(tmp_path):
    calls = []
    identity = {"model": "weights", "tokens": 64, "environment": "pytorch-a"}

    def evaluator(output):
        calls.append(output)
        if len(calls) == 1:
            write_json(output / "partial.json", {"question": 1})
            raise RuntimeError("evaluation interrupted")
        return {"total": 1, "samples": [{"correct": True}], "accuracy": 1.0}

    with pytest.raises(RuntimeError, match="interrupted"):
        cached_evaluate(identity, tmp_path, evaluator)
    result, miss = cached_evaluate(identity, tmp_path, evaluator)
    cached, hit = cached_evaluate(identity, tmp_path, evaluator)
    assert result == cached and hit["hit"] and not miss["hit"] and len(calls) == 2
    assert (calls[0] / "partial.json").exists()
    cached_evaluate({**identity, "environment": "pytorch-b"}, tmp_path, evaluator)
    assert len(calls) == 3
    (tmp_path / hit["key"] / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="changed evidence"):
        cached_evaluate(identity, tmp_path, evaluator)


@pytest.mark.parametrize("change", ["lr", "rank", "missing_limit", "questions", "dataset"])
def test_summary_rejects_incompatible_result(tmp_path, change):
    job = matrix.build_jobs(campaign_args(tmp_path))[0]
    row = result_row(job)
    if change == "lr":
        row["training_settings"]["learning_rate"] = 2e-5
    elif change == "rank":
        row["training_settings"]["lora_rank"] = 16
    elif change == "missing_limit":
        del row["evaluations"]["256"]
    elif change == "questions":
        row["evaluations"]["64"]["after"]["total"] = 32
    else:
        row["dataset_sha256"]["train"] = "other-training-data"
    publish_result(tmp_path, job, row)
    with pytest.raises(ValueError):
        matrix.summarize([job], tmp_path)


def test_summary_requires_sync_pair_and_reports_partial_matrix(tmp_path):
    jobs = matrix.build_jobs(campaign_args(tmp_path))[:2]
    publish_result(tmp_path, jobs[1], result_row(jobs[1]))
    partial = matrix.summarize(jobs, tmp_path)
    assert not partial["complete"]
    assert partial["missing_jobs"] == [jobs[0]["name"]]
    assert partial["missing_sync_baselines"]
    publish_result(tmp_path, jobs[0], result_row(jobs[0]))
    complete = matrix.summarize(jobs, tmp_path)
    assert complete["complete"] and not complete["missing_sync_baselines"]
    assert complete["training_source_shas"] == ["old-tools"]
    assert complete["evaluation_source_shas"] == ["new-tools"]
    with pytest.raises(ValueError, match="duplicate synchronous baseline"):
        matrix.summarize([jobs[0], jobs[0]], tmp_path)


def test_modification_between_completed_evaluations_is_detected(tmp_path):
    jobs = matrix.build_jobs(campaign_args(tmp_path))[:2]
    for index, job in enumerate(jobs):
        row = result_row(job)
        row["evaluator_sha256"] = f"version-{index}"
        publish_result(tmp_path, job, row)
    with pytest.raises(ValueError, match="incompatible evaluator"):
        matrix.summarize(jobs, tmp_path)


def test_modal_deadline_cancels_call_and_confirms_cleanup(tmp_path, monkeypatch):
    cancelled = []

    class Call:
        object_id = "fc-test"

        def get(self, timeout):
            assert 0 < timeout <= 60
            raise TimeoutError("injected preemption/retry deadline")

        def cancel(self, terminate_containers):
            cancelled.append(terminate_containers)

    class Remote:
        def spawn(self, *arguments):
            assert arguments[-1] > time.time()
            return Call()

    class App:
        app_id = "ap-owned"

        @contextlib.contextmanager
        def run(self):
            yield self

    monkeypatch.setattr(modal_control, "stop_owned_app", lambda app_id, name: {
        "app_id": app_id, "confirmed_stopped": True})
    write_json(tmp_path / "request.json", {})
    budget = BudgetLedger(tmp_path / "budget.json", 10.0)
    with pytest.raises(TimeoutError, match="injected"):
        modal_control.invoke(App(), Remote(), (), output=tmp_path, budget=budget,
                             timeout_s=60, gpus=2, app_name="unique-owned-app")
    assert cancelled == [True]
    assert budget.summary()["unconfirmed_attempts"] == []
    assert json.loads((tmp_path / "cleanup.json").read_text())["confirmed_stopped"]


def test_exhausted_budget_never_provisions_an_app(tmp_path):
    class App:
        def run(self):
            pytest.fail("budget exhaustion must be checked before provisioning")

    budget = BudgetLedger(tmp_path / "budget.json", 1.1)
    with pytest.raises(BudgetExceeded):
        modal_control.invoke(App(), None, (), output=tmp_path, budget=budget,
                             timeout_s=1200, gpus=2, app_name="never-started")


def test_checkpoint_evaluation_is_read_only_and_reuses_both_token_limits(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"base-model")
    training = tmp_path / "training"
    checkpoint = training / "checkpoint-55"
    checkpoint.mkdir(parents=True)
    weights = checkpoint / "adapter_model.safetensors"
    weights.write_bytes(b"trained-adapter")
    row = result_row(matrix.build_jobs(campaign_args(tmp_path))[0])
    row.update(label="sync-lag1", checkpoint="checkpoint-55",
               checkpoint_manifest=file_manifest(checkpoint),
               training_code_sha256=training_code_fingerprint(),
               model_sha256=checkpoint_eval.model_manifest(str(model))["sha256"])
    write_json(training / "training-result.json", {"ok": True, "cases": [row]})
    monkeypatch.setattr(checkpoint_eval, "environment_metadata", lambda: {"torch": "fixed-environment"})
    calls = []

    def evaluate(model_path, data_path, output, adapter=None, *, max_new_tokens, **options):
        calls.append((adapter, max_new_tokens))
        return {"total": 1, "samples": [{"prompt": "synthetic", "correct": True, "response_tokens": 1}],
                "accuracy": 1.0, "reload_equal_tensors": 1 if adapter else None}

    monkeypatch.setattr(checkpoint_eval, "evaluate", evaluate)
    base = ["--model-path", str(model), "--training-output", str(training),
            "--cache-dir", str(tmp_path / "cache"), "--eval-max-new-tokens", "64", "256"]
    assert checkpoint_eval.main([*base, "--output-dir", str(tmp_path / "first")]) == 0
    assert checkpoint_eval.main([*base, "--output-dir", str(tmp_path / "retry")]) == 0
    assert calls == [(None, 64), (checkpoint, 64), (None, 256), (checkpoint, 256)]
    assert weights.read_bytes() == b"trained-adapter"
    result = json.loads((tmp_path / "retry/result.json").read_text())["cases"][0]
    assert result["training_source_sha"] == "old-tools"
    assert result["evaluation_source_sha"] != result["training_source_sha"]
    assert result["checkpoint_unchanged"]
