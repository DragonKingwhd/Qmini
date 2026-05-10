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


def check_initial_pose(joints: JointDriver, tol_rad: float = 0.15) -> None:
    """Read current joint positions; warn loudly if far from default pose.

    The policy starts from the default pose; if the robot is not there at
    startup, the first integration steps will diverge. Meant for operator
    confirmation, not a hard error.
    """
    pos, _ = joints.read()
    pos = np.asarray(pos, dtype=np.float32)
    ref = np.asarray(DEFAULT_JOINT_POS_VEC, dtype=np.float32)
    err = np.abs(pos - ref)
    bad = np.where(err > tol_rad)[0]
    if len(bad):
        print(f"[WARN] joints {bad.tolist()} differ from default pose by > {tol_rad} rad")
        print(f"       current: {pos.tolist()}")
        print(f"       default: {ref.tolist()}")
