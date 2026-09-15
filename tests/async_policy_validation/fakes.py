"""Deterministic CPU protocol adapters. No test doubles live in the package."""

from __future__ import annotations

from areno.api.models import RolloutResult, RolloutSequence
from areno.experimental.async_policy import AsyncPrompt, build_batch_envelope


class FakeLifecycle:
    def __init__(self, *, failures=None, gates=None, started_events=None):
        self.failures = dict(failures or {})
        self.gates = dict(gates or {})
        self.started_events = dict(started_events or {})
        self.calls = 0
        self.initializations = 0
        self.closed = False
        self.active = 0

    def initialize(self, *, timeout_s=None):
        self.initializations += 1

    def close(self, *, timeout_s=None):
        assert self.active == 0, "closed an engine while its model was in use"
        self.closed = True

    def enter(self, timeout_s):
        call = self.calls
        self.calls += 1
        self.active += 1
        if call in self.started_events:
            self.started_events[call].set()
        if call in self.gates and not self.gates[call].wait(timeout_s):
            raise TimeoutError("fake engine gate timed out")
        if call in self.failures:
            raise self.failures[call]
        return call


class FakeRolloutEngine(FakeLifecycle):
    def generate(self, prompt, version, *, timeout_s=None):
        try:
            self.enter(timeout_s)
            return RolloutResult(sequences=[
                RolloutSequence(resp_tokens=[3 + index, 5], resp_logprobs=[-0.5, -0.4])
                for index in range(2)
            ])
        finally:
            self.active -= 1


class FakeTrainEngine(FakeLifecycle):
    def __init__(self, *, skip_calls=(), **kwargs):
        super().__init__(**kwargs)
        self.skip_calls = set(skip_calls)
        self.batches = []
        self.versions = []

    def train(self, batch, version, *, timeout_s=None):
        try:
            call = self.enter(timeout_s)
            self.batches.append(batch)
            self.versions.append(version)
            return call not in self.skip_calls
        finally:
            self.active -= 1


class FakeWeightSync(FakeLifecycle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transfers = []

    def transfer(self, plan, *, timeout_s=None):
        try:
            self.enter(timeout_s)
            self.transfers.append((plan.source_version, plan.target_version))
        finally:
            self.active -= 1


def ready_batch(*, version=0, batch_id="batch"):
    prompt = AsyncPrompt(prompt_id="p", input_tokens=(1, 2))
    result = FakeRolloutEngine().generate(prompt, version)
    return build_batch_envelope(
        run_id="test", batch_id=batch_id, epoch=0, policy_version=version,
        prompt_items=[prompt], rollout_results=[result], rewards=[0.0, 1.0], eos_token_id=2,
    )


def sample_reward(record):
    return float(record.metadata["sample_index"])
