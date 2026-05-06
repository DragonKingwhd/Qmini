"""Custom termination conditions for Qmini BIRL locomotion."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def twist_over(
    env: ManagerBasedRLEnv, max_angle: float = 0.7, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when roll or pitch exceeds threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return (torch.abs(roll) > max_angle) | (torch.abs(pitch) > max_angle)


def height_over(
    env: ManagerBasedRLEnv, min_height: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when base height drops below threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < min_height
