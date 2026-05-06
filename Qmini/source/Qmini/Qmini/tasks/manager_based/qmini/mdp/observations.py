"""Custom observation terms for Qmini BIRL locomotion."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_birl_action_term(env: ManagerBasedRLEnv):
    """Helper to retrieve the BIRL action term from the environment."""
    return env.action_manager._terms["birl_action"]


def _get_static_flag(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Compute static flag: 0 when command is near zero, 1 otherwise. Shape: (N, 1)."""
    commands = env.command_manager.get_command("base_velocity")
    cmd_norm = torch.norm(commands[:, :3], dim=1, keepdim=True)
    return (cmd_norm >= 0.15).float()


# ---- Actor Observation Terms (49D total) ----


def velocity_commands_xz(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Velocity commands [x_vel, yaw_rate]. Shape: (N, 2)."""
    commands = env.command_manager.get_command("base_velocity")
    return commands[:, [0, 2]]


def base_euler_rp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base roll and pitch from quaternion. Shape: (N, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.stack([roll, pitch], dim=1)


def base_ang_vel_scaled(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base angular velocity in body frame, scaled by 0.5. Shape: (N, 3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b * 0.5


def joint_pos_rel_to_ref(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions relative to reference pose. Shape: (N, 10)."""
    asset: Articulation = env.scene[asset_cfg.name]
    birl = _get_birl_action_term(env)
    return asset.data.joint_pos - birl.ref_joint_pos


def joint_vel_scaled(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities scaled by 0.1. Shape: (N, 10)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel * 0.1


def joint_pos_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint position error: target - actual. Shape: (N, 10)."""
    asset: Articulation = env.scene[asset_cfg.name]
    birl = _get_birl_action_term(env)
    return birl.current_joint_target - asset.data.joint_pos


def phase_signal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Phase modulator signal: [sin(phase), cos(phase)] * static_flag. Shape: (N, 4)."""
    birl = _get_birl_action_term(env)
    static_flag = _get_static_flag(env)
    return birl.pm_phase * static_flag


def phase_freq_signal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Phase frequency signal: (freq*0.3 - 1) * static_flag. Shape: (N, 4)."""
    birl = _get_birl_action_term(env)
    static_flag = _get_static_flag(env)
    pm_f = birl.pm_frequency
    # pm_f is (N, 2), expand to match pm_phase dimension (N, 4) by repeating
    freq_signal = torch.cat([pm_f * 0.3 - 1.0, pm_f * 0.3 - 1.0], dim=1)
    return freq_signal * static_flag


# ---- Critic Observation Terms (additional privileged info) ----


def cmd_lin_vel_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Linear velocity tracking error. Shape: (N, 1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command("base_velocity")
    return commands[:, [0]] - asset.data.root_lin_vel_b[:, [0]]


def cmd_ang_vel_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Angular velocity tracking error. Shape: (N, 1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command("base_velocity")
    return commands[:, [2]] - asset.data.root_ang_vel_b[:, [2]]


def base_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base linear velocity in body frame. Shape: (N, 3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def base_euler_rp_privileged(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Base euler (roll, pitch) - privileged (no delay). Shape: (N, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.stack([roll, pitch], dim=1)


def base_ang_vel_scaled_privileged(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Base angular velocity * 0.5 - privileged. Shape: (N, 3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b * 0.5


def joint_pos_rel_to_ref_privileged(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint pos - ref_pos (privileged, no delay). Shape: (N, 10)."""
    asset: Articulation = env.scene[asset_cfg.name]
    birl = _get_birl_action_term(env)
    return asset.data.joint_pos - birl.ref_joint_pos


def joint_vel_scaled_privileged(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint velocities * 0.1 - privileged. Shape: (N, 10)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel * 0.1


def action_target_rel_to_ref(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Action target positions relative to reference. Shape: (N, 10)."""
    birl = _get_birl_action_term(env)
    return birl.current_joint_target - birl.ref_joint_pos


def joint_pos_error_privileged(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint position error (privileged). Shape: (N, 10)."""
    asset: Articulation = env.scene[asset_cfg.name]
    birl = _get_birl_action_term(env)
    return birl.current_joint_target - asset.data.joint_pos


def last_net_out_joints(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Last network output (joint part only) / 15. Shape: (N, 10)."""
    birl = _get_birl_action_term(env)
    return birl.net_out_history[-1][:, birl.num_legs:] / 15.0


def foot_height_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Foot height (z position of foot links). Shape: (N, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_body_ids = asset.find_bodies(["ankle_pitch_l", "ankle_pitch_r"])[0]
    foot_pos = asset.data.body_pos_w[:, foot_body_ids, 2]
    return torch.clamp(foot_pos, min=-0.5, max=0.5) * 10.0


def base_height_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base height relative to 0.4m. Shape: (N, 1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, [2]] - 0.4) * 10.0


def foot_vel_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Foot linear velocities [vx_l, vy_l, vz_l, vx_r, vy_r, vz_r]. Shape: (N, 6)."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_body_ids = asset.find_bodies(["ankle_pitch_l", "ankle_pitch_r"])[0]
    foot_vel = asset.data.body_lin_vel_w[:, foot_body_ids, :]  # (N, 2, 3)
    return torch.clamp(foot_vel.reshape(-1, 6), min=-8.0, max=8.0) * 0.5


def base_acc_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base acceleration (approximated from velocity change). Shape: (N, 3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    # Use body acceleration if available, otherwise approximate
    acc = asset.data.body_acc_w[:, 0, :3]
    return torch.clamp(acc, min=-20.0, max=20.0) * 0.2


def foot_force_obs(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    """Foot contact forces. Shape: (N, 2)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norms = torch.norm(forces, dim=-1)
    return torch.clamp(force_norms, min=0.0, max=200.0) * 0.01
