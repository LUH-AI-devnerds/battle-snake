"""Save and load trained policy checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import torch
import torch.nn as nn

T = TypeVar("T", bound=nn.Module)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    meta: Dict[str, Any],
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "meta": meta,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    sidecar = path.with_suffix(".meta.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(
    path: str | Path,
    model_factory: Callable[[Dict[str, Any]], T],
    *,
    device: Optional[torch.device] = None,
    load_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[T, Dict[str, Any], Optional[Dict[str, torch.Tensor]]]:
    """
    Load checkpoint. ``model_factory`` receives metadata and must return an initialized model.
    """
    path = Path(path)
    try:
        payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device or "cpu")
    meta = dict(payload["meta"])
    model = model_factory(meta)
    model.load_state_dict(payload["model_state_dict"])
    if device is not None:
        model.to(device)
    opt_state = None
    if load_optimizer and optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        opt_state = payload["optimizer_state_dict"]
    return model, meta, opt_state


def default_checkpoint_dir(log_dir: str) -> Path:
    return Path(log_dir) / "checkpoints"
