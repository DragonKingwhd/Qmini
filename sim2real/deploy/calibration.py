"""Startup-time calibration: IMU zero-bias + initial joint pose check."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .constants import DEFAULT_JOINT_POS, NUM_JOINTS
from .io.interfaces import IMUDriver, JointDriver


@dataclass
class Calibration:
    imu_bias_rp: tuple[float, float] = (0.0, 0.0)         # (roll, pitch) rad
    imu_bias_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    joint_offset: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS, dtype=np.float32))


def load_yaml_config(yaml_path: str | Path) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f) or {}


def calibrate_imu_bias(
    imu: IMUDriver,
    duration_s: float = 3.0,
    dt_s: float = 0.01,
    max_std_rad: float = 0.02,
) -> tuple[tuple[float, float], np.ndarray]:
    """Robot stationary on flat ground; compute mean roll/pitch and gyro bias."""
    rolls: list[float] = []
    pitches: list[float] = []
    gyros: list[np.ndarray] = []
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < duration_s:
        roll, pitch, gyro = imu.read()
        rolls.append(roll)
        pitches.append(pitch)
        gyros.append(np.asarray(gyro, dtype=np.float32))
        time.sleep(dt_s)
    sr = float(np.std(rolls))
    sp = float(np.std(pitches))
    if sr > max_std_rad or sp > max_std_rad:
        raise RuntimeError(
            f"IMU not static during bias calibration (std roll={sr:.4f} rad, "
            f"pitch={sp:.4f} rad, max={max_std_rad})"
        )
    return (float(np.mean(rolls)), float(np.mean(pitches))), np.mean(gyros, axis=0).astype(np.float32)


def check_initial_pose(
    joints: JointDriver,
    tol_rad: float = 0.15,
) -> None:
    """Read current joint positions; warn loudly if far from DEFAULT_JOINT_POS.

    The policy starts from the default pose; if the robot is not there at
    startup, the first integration steps will diverge. Meant for operator
    confirmation, not a hard error.
    """
    pos, _ = joints.read()
    pos = np.asarray(pos, dtype=np.float32)
    ref = np.asarray(DEFAULT_JOINT_POS, dtype=np.float32)
    err = np.abs(pos - ref)
    bad = np.where(err > tol_rad)[0]
    if len(bad):
        print(f"[WARN] joints {bad.tolist()} differ from default pose by > {tol_rad} rad")
        print(f"       current: {pos.tolist()}")
        print(f"       default: {ref.tolist()}")
