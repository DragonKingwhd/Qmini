"""Real hardware drivers for Qmini on Raspberry Pi.

UnitreeJointDriver: 10 GO-M8010-6 motors across 4 USB-serial buses,
talked to via the unitree_actuator_sdk Python bindings.

Mapping (set 2026-05-10 by manual identification on Qmini):
    hip_yaw_l     /dev/ttyUSB0  ID=1
    hip_roll_l    /dev/ttyUSB1  ID=1
    hip_pitch_l   /dev/ttyUSB3  ID=0
    knee_pitch_l  /dev/ttyUSB3  ID=1
    ankle_pitch_l /dev/ttyUSB3  ID=2
    hip_yaw_r     /dev/ttyUSB0  ID=2
    hip_roll_r    /dev/ttyUSB1  ID=0
    hip_pitch_r   /dev/ttyUSB2  ID=0
    knee_pitch_r  /dev/ttyUSB2  ID=1
    ankle_pitch_r /dev/ttyUSB2  ID=2

NOT controlled here: head motor at /dev/ttyUSB0 ID=0. Hold/free it
externally.

Sign and motor-side zero offset must be set by a separate calibration
step and stored in YAML; defaults assume sign=+1 and zero=0.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from ..constants import DEFAULT_JOINT_POS_VEC, JOINT_NAMES, NUM_JOINTS
from .interfaces import JointDriver

_SDK_LIB = "/home/pi/unitree_actuator_sdk/lib"
if _SDK_LIB not in sys.path:
    sys.path.insert(0, _SDK_LIB)

from unitree_actuator_sdk import (  # type: ignore  # noqa: E402
    MotorCmd,
    MotorData,
    MotorMode,
    MotorType,
    SerialPort,
    queryMotorMode,
)

GEAR_RATIO = 6.33
MOTOR_TYPE = MotorType.GO_M8010_6
G2 = GEAR_RATIO * GEAR_RATIO

# Empirically validated motor-side gains for GO-M8010-6 clones (2026-05-10).
# pid_sweep.py on USB3 ID=1 (knee) showed:
#   - kd ≥ 0.5 always triggers self-oscillation → over-current fault
#   - kp ≤ 0.10 with kd ≤ 0.20 is too soft (gear-box stiction wins)
#   - kp = 0.80, kd = 0.10 tracks ~88% with no faults
#
# These are MOTOR-side gains; sim training used joint-side kp 55-105 / kd 0.3-2.5,
# i.e. expected motor-side ≈ 1.4-2.6 / 0.008-0.06. The clones can't do that
# without faulting; expect a softer-than-sim feel and rely on the policy's
# robustness to PD mismatch (it observes actual joint pos in obs).
KP_MOTOR_PREFIX: Dict[str, float] = {
    "hip_yaw":   0.80,
    "hip_roll":  0.80,
    "hip_pitch": 0.80,
    "knee":      0.80,
    "ankle":     0.80,
}
KD_MOTOR_PREFIX: Dict[str, float] = {
    "hip_yaw":   0.10,
    "hip_roll":  0.10,
    "hip_pitch": 0.10,
    "knee":      0.10,
    "ankle":     0.10,
}


def _gain_for(joint: str, table: Dict[str, float]) -> float:
    for prefix, val in table.items():
        if joint.startswith(prefix):
            return val
    raise KeyError(f"no PD gain for {joint}")


@dataclass
class MotorMap:
    port: str
    motor_id: int
    sign: float = 1.0
    zero_motor_rad: float = 0.0


DEFAULT_MOTOR_MAP: Dict[str, MotorMap] = {
    "hip_yaw_l":     MotorMap(port="/dev/ttyUSB0", motor_id=1),
    "hip_roll_l":    MotorMap(port="/dev/ttyUSB1", motor_id=1),
    "hip_pitch_l":   MotorMap(port="/dev/ttyUSB3", motor_id=0),
    "knee_pitch_l":  MotorMap(port="/dev/ttyUSB3", motor_id=1),
    "ankle_pitch_l": MotorMap(port="/dev/ttyUSB3", motor_id=2),
    "hip_yaw_r":     MotorMap(port="/dev/ttyUSB0", motor_id=2),
    "hip_roll_r":    MotorMap(port="/dev/ttyUSB1", motor_id=0),
    "hip_pitch_r":   MotorMap(port="/dev/ttyUSB2", motor_id=0),
    "knee_pitch_r":  MotorMap(port="/dev/ttyUSB2", motor_id=1),
    "ankle_pitch_r": MotorMap(port="/dev/ttyUSB2", motor_id=2),
}


class UnitreeJointDriver(JointDriver):
    """Drive 10 GO-M8010-6 joints in canonical JOINT_NAMES order."""

    def __init__(
        self,
        mapping: Dict[str, MotorMap] | None = None,
        zero_offset_yaml: str | Path | None = None,
        max_target_step_rad: float = 0.30,
    ) -> None:
        if mapping is None:
            mapping = DEFAULT_MOTOR_MAP
        self._maps: List[MotorMap] = [mapping[n] for n in JOINT_NAMES]
        self._signs = np.array([m.sign for m in self._maps], dtype=np.float32)
        self._zeros = np.array([m.zero_motor_rad for m in self._maps], dtype=np.float32)
        self._kp = np.array([_gain_for(n, KP_MOTOR_PREFIX) for n in JOINT_NAMES],
                            dtype=np.float32)
        self._kd = np.array([_gain_for(n, KD_MOTOR_PREFIX) for n in JOINT_NAMES],
                            dtype=np.float32)
        self._max_step = float(max_target_step_rad)

        if zero_offset_yaml is not None:
            self._load_calibration(zero_offset_yaml)

        ports = sorted({m.port for m in self._maps})
        self._serials: Dict[str, SerialPort] = {p: SerialPort(p) for p in ports}

        default_joint = np.asarray(DEFAULT_JOINT_POS_VEC, dtype=np.float32)
        self._last_motor_cmd = self._joint_to_motor(default_joint)
        self._cached_motor_q = self._last_motor_cmd.copy()
        self._cached_motor_dq = np.zeros(NUM_JOINTS, dtype=np.float32)

    # ---- calibration ----
    def _load_calibration(self, path: str | Path) -> None:
        """Read joints.motor_zero_rad and joints.sign as length-10 lists
        ordered like JOINT_NAMES (matching joints.offset convention)."""
        cfg = yaml.safe_load(Path(path).read_text()) or {}
        joints_cfg = cfg.get("joints", {}) or {}
        zeros = joints_cfg.get("motor_zero_rad")
        if zeros is not None:
            arr = np.asarray(zeros, dtype=np.float32)
            if arr.shape != (NUM_JOINTS,):
                raise ValueError(
                    f"joints.motor_zero_rad must have length {NUM_JOINTS}, got {arr.shape}"
                )
            self._zeros = arr
        signs = joints_cfg.get("sign")
        if signs is not None:
            arr = np.asarray(signs, dtype=np.float32)
            if arr.shape != (NUM_JOINTS,):
                raise ValueError(
                    f"joints.sign must have length {NUM_JOINTS}, got {arr.shape}"
                )
            self._signs = arr

    # ---- unit conversion ----
    def _joint_to_motor(self, q_joint: np.ndarray) -> np.ndarray:
        return self._zeros + self._signs * q_joint * GEAR_RATIO

    def _motor_to_joint_q(self, q_motor: np.ndarray) -> np.ndarray:
        return (q_motor - self._zeros) * self._signs / GEAR_RATIO

    def _motor_to_joint_dq(self, dq_motor: np.ndarray) -> np.ndarray:
        return dq_motor * self._signs / GEAR_RATIO

    # ---- single-motor I/O ----
    def _send_one(self, idx: int, motor_q: float, kp: float, kd: float) -> Tuple[float, float]:
        m = self._maps[idx]
        cmd = MotorCmd()
        cmd.motorType = MOTOR_TYPE
        cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
        cmd.id = m.motor_id
        cmd.q = float(motor_q)
        cmd.dq = 0.0
        cmd.tau = 0.0
        cmd.kp = float(kp)
        cmd.kd = float(kd)
        data = MotorData()
        data.motorType = MOTOR_TYPE
        self._serials[m.port].sendRecv(cmd, data)
        return float(data.q), float(data.dq)

    # ---- JointDriver API ----
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        for i in range(NUM_JOINTS):
            q, dq = self._send_one(i, self._last_motor_cmd[i], self._kp[i], self._kd[i])
            self._cached_motor_q[i] = q
            self._cached_motor_dq[i] = dq
        return (
            self._motor_to_joint_q(self._cached_motor_q).astype(np.float32),
            self._motor_to_joint_dq(self._cached_motor_dq).astype(np.float32),
        )

    def send_position(self, target_rad: np.ndarray) -> None:
        target = np.asarray(target_rad, dtype=np.float32).reshape(NUM_JOINTS)
        # Per-step rate limit: clamp |new - last_joint_target| ≤ max_step.
        last_joint = self._motor_to_joint_q(self._last_motor_cmd)
        delta = np.clip(target - last_joint, -self._max_step, self._max_step)
        target_clamped = (last_joint + delta).astype(np.float32)
        motor_target = self._joint_to_motor(target_clamped)
        for i in range(NUM_JOINTS):
            q, dq = self._send_one(i, motor_target[i], self._kp[i], self._kd[i])
            self._cached_motor_q[i] = q
            self._cached_motor_dq[i] = dq
        self._last_motor_cmd = motor_target.copy()

    def emergency_stop(self) -> None:
        """Drop torque on all 10 joints (kp=kd=0). Head motor not touched."""
        for i in range(NUM_JOINTS):
            try:
                self._send_one(i, self._cached_motor_q[i], 0.0, 0.0)
            except Exception as e:
                print(f"[ESTOP] joint {JOINT_NAMES[i]}: {e!r}")


# ---- helper for the calibration step (separate from the driver) ----
@dataclass
class CalibrationCapture:
    """Reads motor-side q for all 10 joints with kp=kd=0 (zero-torque)."""
    mapping: Dict[str, MotorMap] = field(default_factory=lambda: DEFAULT_MOTOR_MAP)

    def capture_zero_pose(self) -> Dict[str, float]:
        """User holds the robot in default joint pose, then we snapshot motor q."""
        ports = sorted({m.port for m in self.mapping.values()})
        serials = {p: SerialPort(p) for p in ports}
        out: Dict[str, float] = {}
        for name in JOINT_NAMES:
            m = self.mapping[name]
            cmd = MotorCmd()
            cmd.motorType = MOTOR_TYPE
            cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
            cmd.id = m.motor_id
            cmd.q = 0.0
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = 0.0
            cmd.kd = 0.0
            data = MotorData()
            data.motorType = MOTOR_TYPE
            serials[m.port].sendRecv(cmd, data)
            out[name] = float(data.q)
        return out
