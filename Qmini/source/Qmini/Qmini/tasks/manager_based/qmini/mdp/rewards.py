"""Reward terms for the Qmini velocity walking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _yaw_frame_vel_x(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    return vel_yaw[:, 0]


def track_lin_vel_x_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    vel_x = _yaw_frame_vel_x(env, asset_cfg)
    return torch.exp(-((vel_x - cmd_x) ** 2) / (2.0 * std**2))


def forward_progress_yaw_frame(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return torch.clamp(_yaw_frame_vel_x(env, asset_cfg), min=0.0)


def backward_velocity_penalty_yaw_frame(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return torch.clamp(-_yaw_frame_vel_x(env, asset_cfg), min=0.0)


def stall_penalty_yaw_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    vel_x = torch.clamp(_yaw_frame_vel_x(env, asset_cfg), min=0.0)
    return torch.clamp(threshold - vel_x, min=0.0) * (cmd_x > 0.01).float()


def upright_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    tilt_sq = asset.data.projected_gravity_b[:, 0] ** 2 + asset.data.projected_gravity_b[:, 1] ** 2
    return torch.exp(-4.0 * tilt_sq)


def height_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.exp(-18.0 * (asset.data.root_pos_w[:, 2] - target_height) ** 2)


def hip_alternation_reward(
    env: ManagerBasedRLEnv,
    left_hip_name: str,
    right_hip_name: str,
    target_separation: float,
    antiphase_sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    left_idx = asset.data.joint_names.index(left_hip_name)
    right_idx = asset.data.joint_names.index(right_hip_name)
    left = asset.data.joint_pos[:, left_idx]
    right = -asset.data.joint_pos[:, right_idx]
    separation = torch.abs(left - right)
    separation_reward = torch.clamp(separation / target_separation, max=1.0)
    antiphase_reward = torch.exp(-((left + right) ** 2) / (2.0 * antiphase_sigma**2))
    return separation_reward * antiphase_reward


def knee_flexion_reward(
    env: ManagerBasedRLEnv,
    left_knee_name: str,
    right_knee_name: str,
    target: float,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    left_idx = asset.data.joint_names.index(left_knee_name)
    right_idx = asset.data.joint_names.index(right_knee_name)
    left = torch.abs(asset.data.joint_pos[:, left_idx])
    right = torch.abs(asset.data.joint_pos[:, right_idx])
    return torch.exp(-((0.5 * (left + right) - target) ** 2) / (2.0 * sigma**2))


def knee_symmetry_penalty(
    env: ManagerBasedRLEnv,
    left_knee_name: str,
    right_knee_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    left_idx = asset.data.joint_names.index(left_knee_name)
    right_idx = asset.data.joint_names.index(right_knee_name)
    return torch.square(asset.data.joint_pos[:, left_idx] + asset.data.joint_pos[:, right_idx])


def hip_symmetry_penalty(
    env: ManagerBasedRLEnv,
    left_hip_name: str,
    right_hip_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    left_idx = asset.data.joint_names.index(left_hip_name)
    right_idx = asset.data.joint_names.index(right_hip_name)
    return torch.square(asset.data.joint_pos[:, left_idx] + asset.data.joint_pos[:, right_idx])


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def lin_vel_y_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    )
    asset = env.scene[asset_cfg.name]
    foot_vel_xy = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    return torch.sum(foot_vel_xy.norm(dim=-1) * contacts, dim=1)


def base_height_below_threshold(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height
