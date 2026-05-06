"""Custom reward functions for Qmini BIRL locomotion.

All reward functions access shared state computed in the custom environment class
(QminiBIRLEnv) via env.birl_state dict. Each function returns a (num_envs,) tensor.

Note: In Isaac Lab, the reward manager computes: function_return * weight * dt.
The source code applies per-term clipping clip(value * dt, -4, 5). We apply clipping
inside each function where the weight already includes the source weight, and let
Isaac Lab multiply by dt. Total reward clipping to min=0 is done in the env subclass.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_birl(env):
    """Get the BIRL action term."""
    return env.action_manager._terms["birl_action"]


def _get_shared(env):
    """Get shared BIRL state dict computed in env.step()."""
    return env.birl_state


# =============================================================================
# Reward functions
# =============================================================================


def constant_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant reward = 1.0. Weight: 0.3."""
    return torch.ones(env.num_envs, device=env.device)


def base_height_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base height reward: exp(-70 * (z - 0.45)^2). Weight: 1.0 (included in computation)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.exp(-70.0 * (asset.data.root_pos_w[:, 2] - 0.45) ** 2)


def balance_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Balance reward combining height and orientation. Weight: 1.5."""
    s = _get_shared(env)
    return s["balance_rew"].squeeze(-1)


def forward_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Forward velocity tracking. Weight: 2.3."""
    s = _get_shared(env)
    asset: Articulation = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command("base_velocity")
    lin_vel_x_norm = s["lin_vel_x_norm"]
    return torch.exp(
        -torch.clamp(5.0 / lin_vel_x_norm, min=2.0, max=10.0) * (commands[:, [0]] - asset.data.root_lin_vel_b[:, [0]]) ** 2
    ).squeeze(-1)


def lateral_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Lateral velocity penalty. Weight: 0.7."""
    s = _get_shared(env)
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_x_norm = s["lin_vel_x_norm"]
    static_flag = s["static_flag"]
    rew = torch.exp(
        -torch.clamp(5.0 / lin_vel_x_norm, min=3.0, max=15.0) * torch.norm(asset.data.root_lin_vel_b[:, [1]], dim=1, keepdim=True) ** 2
    )
    rew += -0.6 / lin_vel_x_norm * torch.norm(asset.data.root_lin_vel_b[:, [1]], dim=1, keepdim=True) * static_flag
    return rew.squeeze(-1)


def yaw_rate_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Yaw rate tracking. Weight: 2.5."""
    s = _get_shared(env)
    asset: Articulation = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command("base_velocity")
    yaw_rate_norm = s["yaw_rate_norm"]
    return torch.exp(
        -torch.clamp(2.0 / yaw_rate_norm, min=2.0, max=6.0) * (commands[:, [2]] - asset.data.root_ang_vel_b[:, [2]]) ** 2
    ).squeeze(-1)


def angular_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Angular velocity (roll/pitch) penalty. Weight: 0.6."""
    s = _get_shared(env)
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_x_norm = s["lin_vel_x_norm"]
    return torch.exp(
        -torch.clamp(2.0 / lin_vel_x_norm, min=0.7, max=6.0) * torch.norm(asset.data.root_ang_vel_b[:, :2], dim=1, keepdim=True) ** 2
    ).squeeze(-1)


def vertical_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Vertical velocity penalty. Weight: 0.6."""
    s = _get_shared(env)
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_x_norm = s["lin_vel_x_norm"]
    static_flag = s["static_flag"]
    rew = torch.exp(
        -torch.clamp(5.0 / lin_vel_x_norm, min=2.0, max=10.0) * torch.norm(asset.data.root_lin_vel_b[:, [2]], dim=1, keepdim=True) ** 2
    )
    rew -= 0.2 / lin_vel_x_norm * torch.norm(asset.data.root_lin_vel_b[:, 1:], dim=1, keepdim=True) * static_flag
    return rew.squeeze(-1)


def twist_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Twist (body orientation) penalty. Weight: 2.5."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    euler_rp = torch.stack([roll, pitch], dim=1)
    return -torch.norm(euler_rp, dim=-1)


def base_acceleration_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Base acceleration penalty. Weight: 0.1 * balance_rew."""
    s = _get_shared(env)
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_x_norm = s["lin_vel_x_norm"]
    static_flag = s["static_flag"]
    balance_rew = s["balance_rew"]
    gravity_comp = torch.tensor([0, 0, 9.81], device=env.device)
    acc = asset.data.body_acc_w[:, 0, :3]
    rew = -0.4 / lin_vel_x_norm * torch.norm((acc - gravity_comp) * 0.1, dim=1, keepdim=True) * static_flag
    return (rew * balance_rew).squeeze(-1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Foot clearance: swing feet should be off ground. Weight: 1.0."""
    s = _get_shared(env)
    birl = _get_birl(env)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    foot_frc = torch.norm(forces, dim=-1)
    swing_foot_index = foot_frc < 1.0
    static_flag = s["static_flag"]
    rew = torch.sum(
        torch.logical_and(swing_foot_index, birl.foot_swing_mask).float(), dim=1, keepdim=True
    ) / birl.num_legs
    return (rew * static_flag).squeeze(-1)


def foot_support_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Foot support: stance feet should be on ground. Weight: 0.7."""
    s = _get_shared(env)
    birl = _get_birl(env)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    foot_frc = torch.norm(forces, dim=-1)
    support_foot_index = foot_frc >= 10.0
    static_flag = s["static_flag"]
    rew = torch.sum(
        torch.logical_and(support_foot_index, birl.foot_support_mask).float(), dim=1, keepdim=True
    ) / birl.num_legs
    return (rew * static_flag).squeeze(-1)


def foot_height_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Foot height reward for swing phase. Weight: 0.7."""
    s = _get_shared(env)
    birl = _get_birl(env)
    asset: Articulation = env.scene[asset_cfg.name]
    static_flag = s["static_flag"]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    foot_frc = torch.norm(forces, dim=-1)
    support_foot_index = foot_frc >= 10.0

    foot_body_ids = asset.find_bodies(["ankle_pitch_l", "ankle_pitch_r"])[0]
    foot_height = asset.data.body_pos_w[:, foot_body_ids, 2]

    foot_heit_score = 40.0 * torch.clamp(foot_height, min=0.0, max=0.05)
    rew = torch.sum(birl.foot_swing_mask * foot_heit_score, dim=1, keepdim=True).clamp(max=2.0) * static_flag
    rew += -20.0 * torch.sum((foot_height - 0.06).clamp(min=0.0), dim=1, keepdim=True)
    rew += -0.2 * torch.sum(birl.foot_support_mask * foot_heit_score, dim=1, keepdim=True) * static_flag
    rew += -0.2 * torch.sum(support_foot_index * foot_heit_score, dim=1, keepdim=True) * static_flag
    return rew.squeeze(-1)


def foot_soft_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Smooth foot contact forces. Weight: 2.7 * balance_rew."""
    s = _get_shared(env)
    birl = _get_birl(env)
    lin_vel_x_norm = s["lin_vel_x_norm"]
    balance_rew = s["balance_rew"]

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    foot_frc = torch.norm(forces, dim=-1)

    foot_frc_acc = foot_frc - birl.last_foot_frc
    rew = -0.1 * torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=1.5) * torch.norm(foot_frc_acc, dim=1, keepdim=True) / 100.0

    # Update last foot force
    birl.last_foot_frc = foot_frc.clone().detach()

    return (rew * balance_rew).squeeze(-1)


def feet_contact_force_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Contact force penalty. Weight: 0.001."""
    s = _get_shared(env)
    birl = _get_birl(env)
    static_flag = s["static_flag"]

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    foot_frc = torch.norm(forces, dim=-1)
    support_foot_index = foot_frc >= 10.0

    rew = -torch.norm(foot_frc * birl.foot_swing_mask, dim=1, keepdim=True) * static_flag
    rew += -torch.norm((torch.abs(foot_frc - 55.0) * support_foot_index).clamp(min=0.0), dim=1, keepdim=True)
    return rew.squeeze(-1)


def foot_slip_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Foot slip penalty. Weight: 0.5 * balance_rew."""
    s = _get_shared(env)
    birl = _get_birl(env)
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_x_norm = s["lin_vel_x_norm"]
    static_flag = s["static_flag"]
    balance_rew = s["balance_rew"]
    commands = env.command_manager.get_command("base_velocity")

    foot_body_ids = asset.find_bodies(["ankle_pitch_l", "ankle_pitch_r"])[0]
    foot_vel = asset.data.body_lin_vel_w[:, foot_body_ids, :]  # (N, 2, 3)
    foot_height = asset.data.body_pos_w[:, foot_body_ids, 2]
    clip_foot_h = torch.abs(foot_height) + 0.03

    # Swing foot forward velocity bonus
    rew = 2.0 * (lin_vel_x_norm * torch.sum(
        foot_vel[:, :, 0] * commands[:, [0]].sign() * birl.foot_swing_mask, dim=1, keepdim=True
    )).clamp(min=0.0, max=1.0) * static_flag

    # Lateral velocity penalty
    rew += -0.5 * torch.norm(torch.norm(foot_vel[:, :, [1]], dim=-1), dim=1, keepdim=True) * static_flag

    # Static foot movement penalty
    foot_vel_xy = torch.norm(foot_vel[:, :, :2], dim=-1)
    rew += 0.3 * torch.norm(foot_vel_xy, dim=1, keepdim=True) * (static_flag - 1.0)

    # Support foot slip penalty
    rew += -0.3 / lin_vel_x_norm * torch.norm(
        0.1 * foot_vel_xy / clip_foot_h * birl.foot_support_mask, dim=1, keepdim=True
    ) * static_flag

    return (rew * balance_rew).squeeze(-1)


def foot_vertical_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Foot vertical velocity penalty. Weight: 0.2 * balance_rew."""
    s = _get_shared(env)
    birl = _get_birl(env)
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_x_norm = s["lin_vel_x_norm"]
    static_flag = s["static_flag"]
    balance_rew = s["balance_rew"]

    foot_body_ids = asset.find_bodies(["ankle_pitch_l", "ankle_pitch_r"])[0]
    foot_vel = asset.data.body_lin_vel_w[:, foot_body_ids, :]  # (N, 2, 3)
    foot_height = asset.data.body_pos_w[:, foot_body_ids, 2]
    clip_foot_h = torch.abs(foot_height) + 0.03

    foot_vz = foot_vel[:, :, 2].clamp(max=0.0)  # Only negative (downward)
    rew = -0.1 * torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=1.0) * torch.norm(
        torch.norm(foot_vz.unsqueeze(-1), dim=-1) / clip_foot_h, dim=1, keepdim=True
    ) * static_flag
    rew += 0.8 * torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=1.0) * torch.norm(
        torch.norm(foot_vz.unsqueeze(-1), dim=-1), dim=1, keepdim=True
    ) * (static_flag - 1.0)

    return (rew * balance_rew).squeeze(-1)


def foot_acceleration_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Foot acceleration penalty. Weight: 0.05 * balance_rew."""
    s = _get_shared(env)
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_x_norm = s["lin_vel_x_norm"]
    balance_rew = s["balance_rew"]

    foot_body_ids = asset.find_bodies(["ankle_pitch_l", "ankle_pitch_r"])[0]
    foot_vel = asset.data.body_lin_vel_w[:, foot_body_ids, :]
    foot_vz = foot_vel[:, :, 2]  # (N, 2)

    rew = -0.4 * torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=2.0) * torch.norm(foot_vz, dim=1, keepdim=True)
    return (rew * balance_rew).squeeze(-1)


def action_smoothness_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Action smoothness (jerk penalty). Weight: 1.5 * balance_rew."""
    s = _get_shared(env)
    birl = _get_birl(env)
    lin_vel_x_norm = s["lin_vel_x_norm"]
    balance_rew = s["balance_rew"]
    hist = birl.action_history
    jerk = hist[-3] - 2.0 * hist[-2] + hist[-1]
    rew = -0.3 * torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=2.0) * torch.norm(jerk, dim=1, keepdim=True)
    return (rew * balance_rew).squeeze(-1)


def net_out_smoothness_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Network output smoothness. Weight: 0.001 * balance_rew."""
    s = _get_shared(env)
    birl = _get_birl(env)
    lin_vel_x_norm = s["lin_vel_x_norm"]
    balance_rew = s["balance_rew"]
    hist = birl.net_out_history
    jerk = (hist[-3] - 2.0 * hist[-2] + hist[-1])[:, birl.num_legs:]
    rew = -0.2 * torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=2.0) * torch.norm(jerk, dim=1, keepdim=True) ** 2
    return (rew * balance_rew).squeeze(-1)


def action_constraint_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Action constraint: penalize deviation from reference. Weight: 0.2 * balance_rew."""
    s = _get_shared(env)
    birl = _get_birl(env)
    lin_vel_x_norm = s["lin_vel_x_norm"]
    static_flag = s["static_flag"]
    balance_rew = s["balance_rew"]
    diff = birl.current_joint_target - birl.ref_joint_pos
    rew = -0.1 * torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=1.0) * torch.norm(diff, dim=1, keepdim=True)
    rew += -3.0 * torch.norm(diff[:, [0, 1, 5, 6]], dim=1, keepdim=True) * static_flag
    return (rew * balance_rew).squeeze(-1)


def support_ankle_constraint_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Support ankle constraint. Weight: 0.1 * balance_rew."""
    s = _get_shared(env)
    birl = _get_birl(env)
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_x_norm = s["lin_vel_x_norm"]
    static_flag = s["static_flag"]
    balance_rew = s["balance_rew"]

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    foot_frc = torch.norm(forces, dim=-1)
    support_foot_index = foot_frc >= 10.0

    diff = birl.current_joint_target - birl.ref_joint_pos
    rew = -0.1 * torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=1.0) * torch.norm(diff, dim=1, keepdim=True) ** 2 * static_flag
    # Left leg support constraint
    diff_joints = asset.data.joint_pos - birl.ref_joint_pos
    rew += -static_flag * torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=1.0) * torch.norm(
        diff_joints[:, :5] * support_foot_index[:, [0]], dim=1, keepdim=True
    ) ** 2
    # Right leg support constraint
    rew += -static_flag * torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=1.0) * torch.norm(
        diff_joints[:, 5:] * support_foot_index[:, [1]], dim=1, keepdim=True
    ) ** 2
    return (rew * balance_rew).squeeze(-1)


def joint_pos_error_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint position tracking error. Weight: 0.2 * balance_rew."""
    s = _get_shared(env)
    birl = _get_birl(env)
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_x_norm = s["lin_vel_x_norm"]
    balance_rew = s["balance_rew"]
    rew = -0.4 * torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=1.0) * torch.norm(
        birl.current_joint_target - asset.data.joint_pos, dim=1, keepdim=True
    ) ** 2
    return (rew * balance_rew).squeeze(-1)


def joint_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint velocity penalty. Weight: 0.003 * balance_rew."""
    s = _get_shared(env)
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_x_norm = s["lin_vel_x_norm"]
    balance_rew = s["balance_rew"]
    rew = -0.4 * torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=1.0) * torch.norm(
        asset.data.joint_vel, dim=1, keepdim=True
    ) ** 2
    rew += -torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=1.0) * torch.norm(
        asset.data.joint_vel[:, [0, 1, 5, 6]], dim=1, keepdim=True
    ) ** 2
    return (rew * balance_rew).squeeze(-1)


def joint_torque_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint torque exceeding limit penalty. Weight: 0.001."""
    s = _get_shared(env)
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_x_norm = s["lin_vel_x_norm"]
    static_flag = s["static_flag"]
    torque_limits = torch.tensor(
        [20.0, 60.0, 20.0, 20.0, 20.0, 20.0, 60.0, 20.0, 20.0, 20.0],
        device=env.device,
    )
    applied_torque = asset.data.applied_torque
    rew = -0.4 * torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=2.0) * torch.sum(
        (torch.abs(applied_torque) - torque_limits).clamp(min=0.0), dim=1, keepdim=True
    )
    return (rew * static_flag).squeeze(-1)


def phase_freq_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Phase modulator frequency smoothness. Weight: 0.03 * balance_rew."""
    s = _get_shared(env)
    birl = _get_birl(env)
    lin_vel_x_norm = s["lin_vel_x_norm"]
    static_flag = s["static_flag"]
    balance_rew = s["balance_rew"]
    hist = birl.net_out_history
    freq_jerk = (hist[-3] - 2.0 * hist[-2] + hist[-1])[:, :birl.num_legs]

    rew = -0.02 * torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=1.0) * torch.norm(freq_jerk, dim=1, keepdim=True)
    rew += -0.5 * torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=1.0) * torch.norm(
        hist[-1][:, :birl.num_legs] * birl.foot_support_mask, dim=1, keepdim=True
    ) ** 2
    return (rew * static_flag * balance_rew).squeeze(-1)


def net_out_value_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Network output magnitude regularization. Weight: 0.0001 * balance_rew."""
    s = _get_shared(env)
    birl = _get_birl(env)
    lin_vel_x_norm = s["lin_vel_x_norm"]
    balance_rew = s["balance_rew"]
    rew = -0.4 * torch.clamp(1.0 / lin_vel_x_norm, min=0.0, max=1.0) * torch.norm(
        birl.net_out_history[-1][:, birl.num_legs:], dim=1, keepdim=True
    ) ** 2
    return (rew * balance_rew).squeeze(-1)


def foot_pitch_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Foot pitch angle penalty. Weight: 0.5 * balance_rew."""
    s = _get_shared(env)
    asset: Articulation = env.scene[asset_cfg.name]
    balance_rew = s["balance_rew"]
    foot_body_ids = asset.find_bodies(["ankle_pitch_l", "ankle_pitch_r"])[0]
    foot_quat = asset.data.body_quat_w[:, foot_body_ids, :]  # (N, 2, 4)
    # Extract pitch from each foot quaternion
    _, pitch_l, _ = euler_xyz_from_quat(foot_quat[:, 0, :])
    _, pitch_r, _ = euler_xyz_from_quat(foot_quat[:, 1, :])
    foot_pitch = torch.stack([pitch_l, pitch_r], dim=1)
    rew = -0.5 * torch.norm(foot_pitch, dim=1, keepdim=True)
    return (rew * balance_rew).squeeze(-1)


def leg_width_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Leg width maintenance (target 0.14m). Weight: 0.5 * balance_rew."""
    s = _get_shared(env)
    asset: Articulation = env.scene[asset_cfg.name]
    balance_rew = s["balance_rew"]
    foot_body_ids = asset.find_bodies(["ankle_pitch_l", "ankle_pitch_r"])[0]
    foot_y = asset.data.body_pos_w[:, foot_body_ids, 1]  # (N, 2)
    base_y = asset.data.root_pos_w[:, 1:2]  # (N, 1)
    rew = -torch.norm(torch.abs(foot_y - base_y) - 0.14, dim=1, keepdim=True)
    return (rew * balance_rew).squeeze(-1)


def foot_phase_coordination_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Biped anti-phase gait coordination. Weight: 0.3 * balance_rew."""
    s = _get_shared(env)
    birl = _get_birl(env)
    static_flag = s["static_flag"]
    balance_rew = s["balance_rew"]
    phase = birl.foot_phase.clone()
    lsin = torch.sin(phase)
    lcos = torch.cos(phase)
    rew = -torch.norm(lsin[:, [0]] + lsin[:, [1]], dim=1, keepdim=True) ** 2
    rew += -torch.norm(lcos[:, [0]] + lcos[:, [1]], dim=1, keepdim=True) ** 2
    return (rew * static_flag * balance_rew).squeeze(-1)
