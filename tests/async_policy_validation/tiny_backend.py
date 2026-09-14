"""Real CPU tensor adapters for the audit, not a production AReno backend.

The tiny causal policy predicts from the previous token. Production AReno
materialization, padded/packed layouts, selected logprobs and GRPO loss are
used; model forward, SGD and transport are explicit CPU test adapters.
"""

from __future__ import annotations

import copy
import hashlib
import threading
import time
from typing import Any

import torch
from torch import nn

from areno.api.backend.cuda.losses import grpo_loss_fn
from areno.api.backend.cuda.training import make_train_pack
from areno.api.models import RolloutResult, RolloutSequence
from areno.engine.runtime.logprobs import packed_next_token_logprobs
from areno.engine.runtime.train_step import _pack_train_data

VOCAB = 17


class TinyPolicy(nn.Module):
    def __init__(self, seed: int = 123):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(VOCAB, 12, device="cpu", dtype=torch.float32)
            self.head = nn.Linear(12, VOCAB, device="cpu", dtype=torch.float32)

    def forward(self, tokens):
        return self.head(torch.tanh(self.embedding(tokens)))


def vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()]).clone()


def gradients(model: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()]).clone()


def fingerprint(model: nn.Module) -> str:
    return hashlib.sha256(vector(model).numpy().tobytes()).hexdigest()


def token_reward(tokens: list[int]) -> float:
    return sum((index + 1) * (token + 1) for index, token in enumerate(tokens)) / 100.0


class Audit:
    def __init__(self):
        self.events: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def record(self, stage: str, action: str, **values):
        with self.lock:
            self.events.append(dict(stage=stage, action=action, timestamp=time.monotonic(), **values))


class TorchRollout:
    def __init__(self, model, snapshots, audit, *, seed=41):
        self.model = model
        self.snapshots = snapshots
        self.audit = audit
        self.seed = seed
        self.version = 0
        self.calls = 0
        self.records = {}
        self.started_events = {}
        self.gates = {}

    def generate(self, prompt, version, *, timeout_s=None):
        call = self.calls
        self.calls += 1
        self.audit.record("rollout", "start", prompt_id=prompt.prompt_id, version=version)
        try:
            if call in self.started_events:
                self.started_events[call].set()
            if call in self.gates and not self.gates[call].wait(timeout_s):
                raise TimeoutError("CPU rollout gate timed out")
            assert version == self.version
            expected = self.snapshots[version]
            assert fingerprint(self.model) == expected, "rollout weights do not match their advertised version"
            generator = torch.Generator(device="cpu").manual_seed(self.seed + int(prompt.prompt_id) * 101)
            sequences = []
            with torch.no_grad():
                for sample in range(2 + int(prompt.prompt_id) % 3):
                    context = list(prompt.input_tokens)
                    response, logprobs = [], []
                    for _ in range(2 + (int(prompt.prompt_id) + sample) % 4):
                        logits = self.model(torch.tensor([context[-1]], dtype=torch.long, device="cpu"))[0]
                        distribution = logits.log_softmax(-1)
                        token = int(torch.multinomial(distribution.exp(), 1, generator=generator))
                        response.append(token)
                        logprobs.append(float(distribution[token]))
                        context.append(token)
                    sequences.append(RolloutSequence(resp_tokens=response, resp_logprobs=logprobs))
            assert fingerprint(self.model) == expected, "rollout weights changed during generation"
            result = RolloutResult(sequences=sequences)
            self.records[prompt.prompt_id] = dict(prompt=prompt, result=result, version=version,
                                                  fingerprint=expected,
                                                  rewards=[token_reward(s.resp_tokens) for s in sequences])
            return result
        finally:
            self.audit.record("rollout", "end", prompt_id=prompt.prompt_id, version=version)

    def reward(self, prompt, sample):
        # The experimental callback only receives (prompt, sample_index).
        # This test-only lookup supplies the actual generated completion reward.
        return self.records[prompt.prompt_id]["rewards"][sample]


def reference_values(model, rows):
    """Independent row-wise forward/mask/advantage construction."""
    selected, mask, advantages, old = [], [], [], []
    for row in rows:
        tokens = torch.tensor(row.tokens, dtype=torch.long, device="cpu")
        logprobs = model(tokens[:-1]).log_softmax(-1).gather(-1, tokens[1:, None]).squeeze(-1)
        selected.append(logprobs)
        for position in range(1, len(row.tokens)):
            enabled = position >= row.prompt_len and (not row.loss_mask or row.loss_mask[position])
            mask.append(enabled)
            advantages.append(row.scalar_advantage if enabled else 0.0)
            old.append(row.logprobs[position])
    return (torch.cat(selected), torch.tensor(mask, dtype=torch.bool),
            torch.tensor(advantages, dtype=torch.float32), torch.tensor(old, dtype=torch.float32))


class TorchTrain:
    def __init__(self, model, snapshots, audit, *, skip_calls=()):
        self.model = model
        self.shadow = copy.deepcopy(model)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.1)
        self.shadow_optimizer = torch.optim.SGD(self.shadow.parameters(), lr=0.05, momentum=0.1)
        self.snapshots = snapshots
        self.audit = audit
        self.calls = 0
        self.updates = 0
        self.skip_calls = set(skip_calls)
        self.records = []
        self.batches = []
        self.started_events = {}
        self.gates = {}

    def train(self, batch, version, *, timeout_s=None):
        call = self.calls
        self.calls += 1
        assert version == self.updates
        self.audit.record("train", "start", batch_id=batch.batch_id, version=version)
        try:
            if call in self.started_events:
                self.started_events[call].set()
            if call in self.gates and not self.gates[call].wait(timeout_s):
                raise TimeoutError("CPU train gate timed out")
            rows = list(batch.sequences)
            before = vector(self.model)
            self.optimizer.zero_grad(set_to_none=True)
            self.shadow_optimizer.zero_grad(set_to_none=True)
            pack = _pack_train_data(make_train_pack(rows))
            logits = self.model(pack["input_ids"])
            logprobs = packed_next_token_logprobs(logits, pack["input_ids"], pack["train_cu_seqlens"])
            loss, _ = grpo_loss_fn(pack, logprobs)
            ref_logprobs, ref_mask, ref_advantages, ref_old = reference_values(self.shadow, rows)
            torch.testing.assert_close(logprobs, ref_logprobs, atol=5e-7, rtol=1e-6)
            torch.testing.assert_close(pack["packed_response_mask"], ref_mask, atol=0, rtol=0)
            torch.testing.assert_close(pack["packed_advantages"], ref_advantages, atol=0, rtol=0)
            torch.testing.assert_close(pack["packed_logprobs"], ref_old, atol=0, rtol=0)
            # Match the existing AReno objective, then test behavior-policy
            # importance ratios separately instead of hiding that distinction.
            ref_ratio = (ref_logprobs - ref_logprobs.detach()).exp()
            ref_loss = -(ref_ratio * ref_advantages * ref_mask).sum() / ref_mask.sum().clamp_min(1)
            torch.testing.assert_close(loss, ref_loss, atol=5e-7, rtol=1e-6)
            assert bool(torch.isfinite(loss))
            loss.backward()
            ref_loss.backward()
            grad = gradients(self.model)
            ref_grad = gradients(self.shadow)
            assert bool(torch.isfinite(grad).all())
            torch.testing.assert_close(grad, ref_grad, atol=5e-7, rtol=1e-5)
            stepped = call not in self.skip_calls and float(grad.norm()) > 1e-10
            if stepped:
                self.optimizer.step()
                self.shadow_optimizer.step()
                self.updates += 1
                self.snapshots[self.updates] = fingerprint(self.model)
            after = vector(self.model)
            torch.testing.assert_close(after, vector(self.shadow), atol=5e-7, rtol=1e-5)
            delta = float((after - before).norm())
            assert delta > 0 if stepped else delta == 0
            self.records.append(dict(batch_id=batch.batch_id, batch_version=batch.policy_version,
                                     version=version, version_after=self.updates, stepped=stepped,
                                     loss=float(loss.detach()), grad_norm=float(grad.norm()), delta=delta,
                                     logprob_error=float((logprobs.detach() - ref_logprobs.detach()).abs().max()),
                                     gradient_error=float((grad - ref_grad).abs().max()),
                                     parameter_error=float((after - vector(self.shadow)).abs().max())))
            self.batches.append(batch)
            self.last_arrays = dict(parameters_before=before.tolist(), parameters_after=after.tolist(),
                                    gradients=grad.tolist(), logprobs=logprobs.detach().tolist(),
                                    mask=ref_mask.tolist(), advantages=ref_advantages.tolist(),
                                    rollout_logprobs=ref_old.tolist(), loss=float(loss.detach()))
            return stepped
        finally:
            self.audit.record("train", "end", batch_id=batch.batch_id, version=version)


class TorchSync:
    def __init__(self, train, rollout, audit, coordinator=None, *, fail_partial=False):
        self.train = train
        self.rollout = rollout
        self.audit = audit
        self.coordinator = coordinator
        self.fail_partial = fail_partial
        self.transfers = []

    def transfer(self, plan, *, timeout_s=None):
        self.audit.record("sync", "start", source=plan.source_version, target=plan.target_version)
        try:
            if self.coordinator is not None:
                assert self.coordinator.sync_active
                assert self.coordinator.active_rollouts == 0 and not self.coordinator.train_active
            assert plan.target_version == self.train.updates
            if self.fail_partial:
                with torch.no_grad():
                    next(self.rollout.model.parameters()).copy_(next(self.train.model.parameters()))
                raise RuntimeError("injected partial tensor transfer")
            self.rollout.model.load_state_dict(self.train.model.state_dict())
            self.rollout.version = plan.target_version
            assert fingerprint(self.rollout.model) == self.train.snapshots[plan.target_version]
            self.transfers.append((plan.source_version, plan.target_version))
        finally:
            self.audit.record("sync", "end", source=plan.source_version, target=plan.target_version)


def engines(coordinator=None, *, seed=123, skip_calls=()):
    torch.set_num_threads(1)
    model = TinyPolicy(seed)
    snapshots = {0: fingerprint(model)}
    audit = Audit()
    train = TorchTrain(model, snapshots, audit, skip_calls=skip_calls)
    rollout = TorchRollout(copy.deepcopy(model), snapshots, audit)
    sync = TorchSync(train, rollout, audit, coordinator)
    return rollout, train, sync, audit
