"""Curriculum terms for Qmini walking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import RLTaskEnv


def velocity_command_curriculum(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    command_name: str,
    min_vel: float,
    max_vel: float,
    success_threshold: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Increase the commanded forward velocity once tracking is reliable."""

    if not hasattr(env, "_qmini_vel_curriculum_max_vel"):
        env._qmini_vel_curriculum_max_vel = min_vel  # type: ignore[attr-defined]
        env._qmini_vel_curriculum_success_buf = torch.zeros(env.num_envs, device=env.device)  # type: ignore[attr-defined]

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    cmd_x = command[env_ids, 0]
    vel_x = asset.data.root_lin_vel_b[env_ids, 0]
    tracking_ok = (torch.abs(vel_x - cmd_x) < 0.05).float()
    env._qmini_vel_curriculum_success_buf[env_ids] = tracking_ok  # type: ignore[attr-defined]

    if env._qmini_vel_curriculum_success_buf.mean().item() >= success_threshold:  # type: ignore[attr-defined]
        env._qmini_vel_curriculum_max_vel = min(  # type: ignore[attr-defined]
            env._qmini_vel_curriculum_max_vel + 0.02, max_vel  # type: ignore[attr-defined]
        )

    cmd_term = env.command_manager.get_term(command_name)
    if hasattr(cmd_term, "cfg") and hasattr(cmd_term.cfg, "ranges"):
        cmd_term.cfg.ranges.lin_vel_x = (min_vel, env._qmini_vel_curriculum_max_vel)  # type: ignore[attr-defined]

    return torch.tensor(env._qmini_vel_curriculum_max_vel, device=env.device)  # type: ignore[attr-defined]
