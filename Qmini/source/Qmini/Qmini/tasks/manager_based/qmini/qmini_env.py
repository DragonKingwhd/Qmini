"""Custom ManagerBasedRLEnv subclass for Qmini BIRL locomotion.

Computes shared state (balance_rew, static_flag, normalization factors)
that multiple reward terms need access to, avoiding redundant computation.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.math import euler_xyz_from_quat


class QminiBIRLEnv(ManagerBasedRLEnv):
    """Qmini BIRL locomotion environment with shared reward state."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        # Initialize shared state dict
        self.birl_state = {}

    def _compute_shared_state(self):
        """Compute shared quantities used by multiple reward functions."""
        commands = self.command_manager.get_command("base_velocity")
        robot = self.scene["robot"]

        # Command normalization factors
        lin_vel_x_norm = torch.clamp(torch.abs(commands[:, [0]]), min=0.3, max=2.0) + 0.2
        yaw_rate_norm = torch.clamp(torch.abs(commands[:, [2]]), min=0.3, max=1.5) + 0.2

        # Static flag: 0 when command is near zero, 1 otherwise
        cmd_norm = torch.norm(commands[:, :3], dim=1, keepdim=True)
        static_flag = (cmd_norm >= 0.15).float()

        # Base height reward component
        base_heit_rew = torch.exp(-70.0 * (robot.data.root_pos_w[:, [2]] - 0.45) ** 2)

        # Balance reward: combines height + orientation
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        euler_rp = torch.stack([roll, pitch], dim=1)
        balance_rew = 0.5 * (
            base_heit_rew * torch.exp(
                -torch.clamp(5.0 / lin_vel_x_norm, min=2.0, max=8.0) * torch.norm(euler_rp, dim=-1, keepdim=True)
            ) + 1.0
        )

        self.birl_state = {
            "lin_vel_x_norm": lin_vel_x_norm,
            "yaw_rate_norm": yaw_rate_norm,
            "static_flag": static_flag,
            "base_heit_rew": base_heit_rew,
            "balance_rew": balance_rew,
        }

    def step(self, action: torch.Tensor):
        """Override step to compute shared state before reward computation."""
        # Process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # Check if we need rendering within physics loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # Physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # Post-step
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # >>> CUSTOM: Compute shared state before rewards <<<
        self._compute_shared_state()

        # Reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        # Clip total reward to min=0 (matching source implementation)
        self.reward_buf = torch.clamp(self.reward_buf, min=0.0)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # Reset terminated environments
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        # Update command
        self.command_manager.compute(dt=self.step_dt)
        # Step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # Compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
