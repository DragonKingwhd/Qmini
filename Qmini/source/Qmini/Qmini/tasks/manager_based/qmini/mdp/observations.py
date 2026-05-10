"""Custom observation terms for Qmini walking."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return reference gait phase as [sin, cos]."""

    action_term = env.action_manager.get_term("joint_pos")
    phase = getattr(action_term, "gait_phase", None)
    if phase is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    angle = 2.0 * torch.pi * phase
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
