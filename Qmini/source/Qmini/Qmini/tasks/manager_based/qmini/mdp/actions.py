"""Reference gait action term for Qmini walking.

The policy outputs residual joint-position commands around a simple sinusoidal
biped reference gait. This gives PPO a walking prior instead of asking it to
discover leg phasing from scratch.
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class QminiReferenceGaitAction(ActionTerm):
    """Joint-position action with an anti-phase walking reference."""

    cfg: QminiReferenceGaitActionCfg
    _asset: Articulation

    def __init__(self, cfg: QminiReferenceGaitActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._asset: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        self._scale = cfg.scale

        self._gait_period = max(cfg.gait_period, env.step_dt)
        self._stance_ratio = min(max(cfg.stance_ratio, 0.05), 0.95)
        self._hip_amp = cfg.hip_pitch_amplitude
        self._knee_amp = cfg.knee_pitch_amplitude
        self._ankle_amp = cfg.ankle_pitch_amplitude
        self._push_off_ankle_scale = cfg.push_off_ankle_scale

        self._gait_phase = torch.zeros(env.num_envs, device=env.device)
        self._raw_actions = torch.zeros(env.num_envs, self._num_joints, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, self._num_joints, device=env.device)

        def _find(name: str) -> int | None:
            for index, joint_name in enumerate(self._joint_names):
                if joint_name == name:
                    return index
            return None

        self._left_hip = _find("hip_pitch_l")
        self._right_hip = _find("hip_pitch_r")
        self._left_knee = _find("knee_pitch_l")
        self._right_knee = _find("knee_pitch_r")
        self._left_ankle = _find("ankle_pitch_l")
        self._right_ankle = _find("ankle_pitch_r")

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def gait_phase(self) -> torch.Tensor:
        return self._gait_phase

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        default_pos = self._asset.data.default_joint_pos[:, self._joint_ids]
        self._processed_actions = default_pos + self._compute_reference_offsets() + actions * self._scale

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._gait_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        self._raw_actions[env_ids] = 0.0

    def _advance_phase(self) -> None:
        self._gait_phase = (self._gait_phase + self._env.step_dt / self._gait_period) % 1.0

    def _gait_profile(self, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sr = self._stance_ratio
        in_stance = phase < sr

        stance_progress = phase / sr
        hip_stance = 1.0 - 2.0 * stance_progress
        knee_stance = 0.15 * torch.sin(math.pi * stance_progress)
        push_off = torch.clamp((stance_progress - 0.60) / 0.40, min=0.0)
        ankle_stance = -0.55 * hip_stance + self._push_off_ankle_scale * push_off

        swing_progress = (phase - sr) / (1.0 - sr)
        hip_swing = -1.0 + 2.0 * swing_progress
        knee_swing = torch.sin(math.pi * swing_progress)
        ankle_swing = -0.35 * hip_swing - 0.20 * torch.sin(math.pi * swing_progress)

        hip = torch.where(in_stance, hip_stance, hip_swing)
        knee = torch.where(in_stance, knee_stance, knee_swing)
        ankle = torch.where(in_stance, ankle_stance, ankle_swing)
        return hip, knee, ankle

    def _compute_reference_offsets(self) -> torch.Tensor:
        self._advance_phase()
        offsets = torch.zeros(self._env.num_envs, self._num_joints, device=self._env.device)

        left_phase = self._gait_phase
        right_phase = (self._gait_phase + 0.5) % 1.0
        left_hip, left_knee, left_ankle = self._gait_profile(left_phase)
        right_hip, right_knee, right_ankle = self._gait_profile(right_phase)

        def _set(index: int | None, value: torch.Tensor) -> None:
            if index is not None:
                offsets[:, index] = value

        # The right-leg pitch joints use mirrored signs in the Qmini default pose.
        _set(self._left_hip, self._hip_amp * left_hip)
        _set(self._right_hip, -self._hip_amp * right_hip)
        _set(self._left_knee, self._knee_amp * left_knee)
        _set(self._right_knee, -self._knee_amp * right_knee)
        _set(self._left_ankle, self._ankle_amp * left_ankle)
        _set(self._right_ankle, -self._ankle_amp * right_ankle)
        return offsets


@configclass
class QminiReferenceGaitActionCfg(ActionTermCfg):
    """Configuration for the Qmini reference gait action."""

    class_type: type = QminiReferenceGaitAction
    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    scale: float = 0.10
    gait_period: float = 0.72
    stance_ratio: float = 0.60
    hip_pitch_amplitude: float = 0.22
    knee_pitch_amplitude: float = 0.24
    ankle_pitch_amplitude: float = 0.14
    push_off_ankle_scale: float = 0.18
