"""Single-rank optimizer/RNG state paired with an already-written checkpoint."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(worker) -> dict:
    from areno.engine.modeling import unwrap_model

    if worker.config.role != "train" or (worker.config.tp_size, worker.config.dp_size) != (1, 1):
        raise ValueError("training-state checkpoints currently require train TP=1 / DP=1")
    return {
        "format": 1,
        "optimizer_class": type(worker.optimizer).__qualname__,
        "optimizer_config": json.loads(json.dumps(asdict(worker.config.optimizer))),
        "parameters": [[name, list(parameter.shape), str(parameter.dtype)]
                       for name, parameter in unwrap_model(worker.model).named_parameters() if parameter.requires_grad],
        "has_lora": worker.adapter_registry is not None,
    }


def save_training_state(worker, path: str) -> dict:
    directory = Path(path)
    metadata = _identity(worker)
    weights = sorted(directory.glob("*.safetensors"))
    if not weights:
        raise ValueError("save model or adapter tensors before optimizer state")
    metadata.update(global_step=worker._global_step, weights={item.name: _digest(item) for item in weights})
    state = {"optimizer": worker.optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state(worker.device) if worker.device.type == "cuda" else None}
    torch.save(state, directory / "training_state.pt")
    metadata["state_sha256"] = _digest(directory / "training_state.pt")
    # This manifest is written last. A partial checkpoint has no valid resume marker.
    (directory / "training_state.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"global_step": worker._global_step, "path": str(directory)}


def load_training_state(worker, path: str) -> dict:
    directory = Path(path)
    metadata = json.loads((directory / "training_state.json").read_text())
    identity = _identity(worker)
    if any(metadata.get(key) != value for key, value in identity.items()):
        raise ValueError("training-state optimizer configuration or parameter layout does not match")
    if type(metadata.get("global_step")) is not int or metadata["global_step"] < 0:
        raise ValueError("training-state global_step must be a nonnegative integer")
    for filename, expected in {**metadata["weights"], "training_state.pt": metadata["state_sha256"]}.items():
        if Path(filename).name != filename or _digest(directory / filename) != expected:
            raise ValueError("training-state checkpoint checksum mismatch")
    state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=True)
    worker.optimizer.load_state_dict(state["optimizer"])
    worker._global_step = metadata["global_step"]
    if worker.adapter_registry is not None:
        worker.adapter_registry.version = worker._global_step
    torch.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], worker.device)
    return {"global_step": worker._global_step, "path": str(directory)}
