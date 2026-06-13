"""Startup-time calibration: load YAML config, sanity-check initial pose."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .constants import DEFAULT_JOINT_POS_VEC, NUM_JOINTS
from .io.interfaces import IMUDriver, JointDriver


@dataclass
class Calibration:
    # Mean gyro reading (rad/s) when robot is stationary.
    imu_gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    # Optional per-joint encoder offset (rad), if hardware zero != URDF zero.
    joint_offset: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS, dtype=np.float32))


def load_yaml_config(yaml_path: str | Path) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f) or {}


def calibrate_imu_gyro(
    imu: IMUDriver,
    duration_s: float = 3.0,
    dt_s: float = 0.01,
    max_std_rad_s: float = 0.05,
) -> np.ndarray:
    """Robot stationary on flat ground; mean of body-frame gyro reading."""
    samples: list[np.ndarray] = []
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < duration_s:
        _, gyro, _ = imu.read()
        samples.append(np.asarray(gyro, dtype=np.float32))
        time.sleep(dt_s)
    arr = np.stack(samples, axis=0)
    std = arr.std(axis=0)
    if np.any(std > max_std_rad_s):
        raise RuntimeError(
            f"IMU not static during gyro calibration (per-axis std {std} rad/s, "
            f"max allowed {max_std_rad_s})"
        )
    return arr.mean(axis=0).astype(np.float32)


# One rotor turn (2π motor-side) seen joint-side through the 6.33 gearbox.
# A startup error of ≈ k×0.99 rad means that motor rebooted/power-cycled
# since calibration (multi-turn count reset) — its motor_zero is stale.
ROTOR_TURN_JOINT_RAD = 2.0 * np.pi / 6.33  # ≈ 0.9926


def check_initial_pose(joints: JointDriver, tol_rad: float = 0.15) -> list[int]:
    """Read current joint positions; return indices far from default pose.

    The policy starts from the default pose; if the robot is not there at
    startup, ramping to DEFAULT can physically drive joints into hard stops
    (overcurrent fault). Caller decides whether to abort.
    """
    pos, _ = joints.read()
    pos = np.asarray(pos, dtype=np.float32)
    ref = np.asarray(DEFAULT_JOINT_POS_VEC, dtype=np.float32)
    err = pos - ref
    bad = np.where(np.abs(err) > tol_rad)[0]
    if len(bad):
        print(f"[WARN] joints {bad.tolist()} differ from default pose by > {tol_rad} rad")
        print(f"       current: {pos.tolist()}")
        print(f"       default: {ref.tolist()}")
        for i in bad:
            turns = err[i] / ROTOR_TURN_JOINT_RAD
            if abs(turns - round(turns)) < 0.2 and round(turns) != 0:
                print(f"       joint {i}: 偏差 {err[i]:+.3f} rad ≈ {round(turns):+d} 圈转子"
                      f" → 该电机断电/重启过,motor_zero 已作废,重跑 calib/stand_zero.py")
    return bad.tolist()
