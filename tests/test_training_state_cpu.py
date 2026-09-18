"""Optimizer continuation and corrupt-checkpoint rejection without CUDA."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from areno.engine.checkpoints.training_state import load_training_state, save_training_state
from areno.engine.config import OptimizerConfig


def worker():
    model = torch.nn.Linear(2, 1)
    return SimpleNamespace(model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
                           config=SimpleNamespace(role="train", tp_size=1, dp_size=1, optimizer=OptimizerConfig(lr=0.01)),
                           adapter_registry=None, _global_step=0, device=torch.device("cpu"))


def update(value):
    value.optimizer.zero_grad()
    loss = (value.model(torch.tensor([[1.0, 2.0], [3.0, -1.0]])) - 0.5).square().mean()
    loss.backward()
    value.optimizer.step()
    value._global_step += 1


def test_training_state_continuation_matches_uninterrupted_adam(tmp_path):
    torch.manual_seed(17)
    original = worker()
    update(original)
    update(original)
    save_file(original.model.state_dict(), str(tmp_path / "model.safetensors"))
    save_training_state(original, str(tmp_path))
    expected_random = torch.rand(5)
    update(original)
    resumed = worker()
    resumed.model.load_state_dict(load_file(str(tmp_path / "model.safetensors")))
    assert load_training_state(resumed, str(tmp_path))["global_step"] == 2
    assert torch.equal(torch.rand(5), expected_random)
    update(resumed)
    assert resumed._global_step == original._global_step == 3
    assert all(torch.equal(value, resumed.model.state_dict()[key]) for key, value in original.model.state_dict().items())


def test_training_state_rejects_mismatched_weights(tmp_path):
    value = worker()
    save_file(value.model.state_dict(), str(tmp_path / "model.safetensors"))
    save_training_state(value, str(tmp_path))
    (tmp_path / "model.safetensors").write_bytes(b"corrupt checkpoint")
    with pytest.raises(ValueError, match="checksum"):
        load_training_state(value, str(tmp_path))


def test_training_state_rejects_changed_optimizer_configuration(tmp_path):
    value = worker()
    save_file(value.model.state_dict(), str(tmp_path / "model.safetensors"))
    save_training_state(value, str(tmp_path))
    value.config.optimizer.lr = 0.02
    with pytest.raises(ValueError, match="configuration"):
        load_training_state(value, str(tmp_path))
