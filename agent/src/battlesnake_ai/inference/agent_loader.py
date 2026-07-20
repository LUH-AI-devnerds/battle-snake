"""Load trained agents from checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn

from battlesnake_ai.models.dqn import DQN
from battlesnake_ai.models.ppo_policy import PPOPolicy
from battlesnake_ai.models.rainbow_dqn import RainbowDQN
from battlesnake_ai.training.checkpoint import load_checkpoint


def build_model_from_meta(meta: dict) -> nn.Module:
    algo = meta.get("algorithm", "dqn")
    in_channels = int(meta["in_channels"])
    num_actions = int(meta.get("num_actions", 4))
    if algo in ("dqn",):
        return DQN(in_channels=in_channels, num_actions=num_actions)
    if algo in ("rainbow",):
        num_atoms = int(meta.get("num_atoms", 51))
        feature_dim = int(meta.get("feature_dim", 64))
        noisy = bool(meta.get("noisy", False))
        v_min = float(meta.get("v_min", -1.0))
        v_max = float(meta.get("v_max", 1.0))
        return RainbowDQN(
            in_channels=in_channels,
            num_actions=num_actions,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            feature_dim=feature_dim,
            noisy=noisy,
        )
    if algo in ("ppo", "ppo_finetune"):
        return PPOPolicy(in_channels=in_channels, num_actions=num_actions)
    raise ValueError(f"Unknown algorithm in checkpoint: {algo}")


def load_agent(
    path: str | Path,
    device: torch.device | None = None,
) -> Tuple[nn.Module, dict]:
    dev = device or torch.device("cpu")

    def factory(meta: dict) -> nn.Module:
        return build_model_from_meta(meta)

    model, meta, _ = load_checkpoint(path, factory, device=dev)
    model.eval()
    return model, meta


def copy_rainbow_backbone_to_ppo(rainbow: RainbowDQN, ppo: PPOPolicy) -> None:
    ppo.backbone.load_state_dict(rainbow.backbone.state_dict())
