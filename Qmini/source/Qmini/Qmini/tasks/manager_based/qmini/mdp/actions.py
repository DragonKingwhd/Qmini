"""Custom BIRL action term with PhaseModulator for Qmini bipedal locomotion."""

from __future__ import annotations

import math
import torch
from collections import deque
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class PhaseModulator:
    """Gait phase generator for bipedal locomotion.

    Maintains per-environment phase state for each leg, updated with network-output frequencies.
    Phase determines support/swing mask for gait coordination.
    """

    def __init__(self, time_step: float, num_envs: int, num_legs: int, device: str):
        self.num_legs = num_legs
        self.num_envs = num_envs
        self.device = device
        self._time_step = time_step
        self._phase = torch.zeros(num_envs, num_legs, dtype=torch.float, device=device)
        self._frequency = torch.ones(num_envs, num_legs, dtype=torch.float, device=device) * 0.5
        self.reset(env_ids=torch.arange(num_envs, device=device))

    def reset(self, env_ids: torch.Tensor):
        """Reset phase to random initial values for given environments."""
        init_phase = torch.rand(len(env_ids), self.num_legs, device=self.device) * 2 * math.pi
        self._phase[env_ids] = init_phase % (2 * math.pi)
        self._frequency[env_ids] = 0.5

    def compute(self, frequency: torch.Tensor) -> torch.Tensor:
        """Update phase based on network-output frequencies."""
        self._frequency = frequency
        self._phase = (self._phase + 2 * math.pi * frequency * self._time_step) % (2 * math.pi)
        return self._phase

    @property
    def frequency(self) -> torch.Tensor:
        return self._frequency

    @property
    def phase(self) -> torch.Tensor:
        return self._phase


class BIRLActionTerm(ActionTerm):
    """Bio-Inspired Rhythmic Locomotion action term.

    Processes 12-dim network output: [2 phase frequencies, 10 joint position deltas].
    Maintains phase modulator state, incremental joint position targets, and action history.
    """

    cfg: BIRLActionTermCfg
    _asset: Articulation

    def __init__(self, cfg: BIRLActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._num_joints = self._asset.num_joints  # 10
        self._num_legs = 2
        self._convert_phi = 1.2 * math.pi

        # Action dimensions: 2 (freq) + 10 (joints) = 12
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # Action scaling: incremental mode ranges
        self._action_low = torch.tensor(
            cfg.inc_low_ranges, dtype=torch.float, device=self.device
        )
        self._action_high = torch.tensor(
            cfg.inc_high_ranges, dtype=torch.float, device=self.device
        )

        # Reference joint positions
        self._ref_joint_pos = torch.tensor(
            cfg.ref_joint_pos, dtype=torch.float, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        # Current joint position targets (initialized to default)
        default_joint_pos = self._asset.data.default_joint_pos[0].clone()
        self._current_joint_target = default_joint_pos.unsqueeze(0).repeat(self.num_envs, 1)

        # Joint position limits from URDF
        joint_pos_limits = self._asset.data.soft_joint_pos_limits[0]
        self._joint_limit_low = joint_pos_limits[:, 0].unsqueeze(0).repeat(self.num_envs, 1)
        self._joint_limit_high = joint_pos_limits[:, 1].unsqueeze(0).repeat(self.num_envs, 1)

        # Phase modulator
        step_dt = env.step_dt  # decimation * sim_dt
        self._phase_modulator = PhaseModulator(
            time_step=step_dt, num_envs=self.num_envs, num_legs=self._num_legs, device=self.device
        )

        # Histories for smoothness rewards
        self._action_history = deque(maxlen=3)
        self._net_out_history = deque(maxlen=3)
        for _ in range(3):
            self._action_history.append(self._current_joint_target.clone())
        for _ in range(3):
            self._net_out_history.append(torch.zeros(self.num_envs, self.action_dim, device=self.device))

        # Foot phase masks
        self._update_phase_masks()

        # Last foot force for soft contact reward
        self._last_foot_frc = torch.zeros(self.num_envs, self._num_legs, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._num_legs + self._num_joints  # 2 + 10 = 12

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    # --- Public accessors for observations and rewards ---

    @property
    def phase_modulator(self) -> PhaseModulator:
        return self._phase_modulator

    @property
    def current_joint_target(self) -> torch.Tensor:
        return self._current_joint_target

    @property
    def ref_joint_pos(self) -> torch.Tensor:
        return self._ref_joint_pos

    @property
    def action_history(self) -> deque:
        return self._action_history

    @property
    def net_out_history(self) -> deque:
        return self._net_out_history

    @property
    def foot_support_mask(self) -> torch.Tensor:
        return self._foot_support_mask

    @property
    def foot_swing_mask(self) -> torch.Tensor:
        return self._foot_swing_mask

    @property
    def foot_phase(self) -> torch.Tensor:
        return self._phase_modulator.phase

    @property
    def pm_phase(self) -> torch.Tensor:
        """Phase signal: [sin(phase_l), sin(phase_r), cos(phase_l), cos(phase_r)]"""
        phase = self._phase_modulator.phase
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)

    @property
    def pm_frequency(self) -> torch.Tensor:
        return self._phase_modulator.frequency.clone()

    @property
    def last_foot_frc(self) -> torch.Tensor:
        return self._last_foot_frc

    @last_foot_frc.setter
    def last_foot_frc(self, value: torch.Tensor):
        self._last_foot_frc = value

    @property
    def convert_phi(self) -> float:
        return self._convert_phi

    @property
    def num_legs(self) -> int:
        return self._num_legs

    def _update_phase_masks(self):
        """Update foot support/swing masks based on current phase."""
        phase = self._phase_modulator.phase
        mask_1 = phase >= 0.0
        mask_2 = phase < self._convert_phi
        self._foot_support_mask = torch.logical_and(mask_1, mask_2)
        self._foot_swing_mask = torch.logical_not(self._foot_support_mask)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset action term for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device)

        # Reset joint targets to default positions
        default_pos = self._asset.data.default_joint_pos[0].clone()
        self._current_joint_target[env_ids] = default_pos.unsqueeze(0)

        # Reset phase modulator
        self._phase_modulator.reset(env_ids)
        self._update_phase_masks()

        # Reset last foot force
        self._last_foot_frc[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        """Process raw network output into joint position targets.

        Args:
            actions: Raw network output of shape (num_envs, 12), values in [-1, 1].
        """
        self._raw_actions[:] = actions

        # Scale from [-1, 1] to [action_low, action_high]
        scaled = 0.5 * (actions + 1.0) * (self._action_high - self._action_low) + self._action_low
        self._net_out_history.append(scaled.clone())

        # Update phase modulator with frequency outputs
        freq = scaled[:, :self._num_legs]
        self._phase_modulator.compute(freq)
        self._update_phase_masks()

        # Incremental joint position update
        joint_deltas = scaled[:, self._num_legs:]
        self._current_joint_target += joint_deltas * self._env.step_dt

        # Clip to joint limits
        self._current_joint_target = torch.clamp(
            self._current_joint_target, self._joint_limit_low, self._joint_limit_high
        )

        # Store in action history
        self._action_history.append(self._current_joint_target.clone())

        # Set processed actions (joint position targets)
        self._processed_actions[:] = self._current_joint_target

    def apply_actions(self):
        """Apply joint position targets to the robot actuators."""
        self._asset.set_joint_position_target(self._processed_actions)


@configclass
class BIRLActionTermCfg(ActionTermCfg):
    """Configuration for the BIRL action term."""

    class_type: type = BIRLActionTerm

    # Incremental action ranges: [freq_l, freq_r, joint_0..joint_9]
    inc_low_ranges: list[float] = [0.5, 0.5] + [-15.0] * 10
    inc_high_ranges: list[float] = [3.5, 3.5] + [15.0] * 10

    # Reference joint positions (default standing pose)
    ref_joint_pos: list[float] = [0.4, -0.1, -1.5, 1.0, -1.3, -0.4, 0.1, 1.5, -1.0, 1.3]
