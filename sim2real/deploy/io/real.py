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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from ..constants import DEFAULT_JOINT_POS_VEC, JOINT_NAMES, NUM_JOINTS
from .interfaces import CommandSource, IMUDriver, JointDriver

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

# Empirically validated motor-side gains for GO-M8010-6 (Qmini, 2026-05-10).
# Two pid_sweep.py runs on USB3 ID=1 (knee) found a stable kp/kd workspace.
# Best combo: kp=1.20, kd=0.10 → moved=0.375 (94% tracking on a 0.4 rad step),
# final_err=-0.019 rad (~1° joint side), max_dev=0.067 rad, zero oscillations.
#
# Kp=1.20 maps to ~48 N·m/rad on the joint side (kp_joint = kp_motor × G²).
# Sim training used 30–105 N·m/rad; some joints (hip_roll, hip_pitch) will be
# softer than sim — relying on policy robustness via obs feedback.
#
# NEVER set kd ≥ 0.50 on this hardware (every test in that region triggered
# 5-rad oscillations and current-fault flash).
KP_MOTOR_PREFIX: Dict[str, float] = {
    "hip_yaw":   1.20,
    "hip_roll":  1.20,
    "hip_pitch": 1.20,
    "knee":      1.20,
    "ankle":     1.20,
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


# =============================================================================
# IMU driver — GY-91 (MPU9250 + BMP280) over I2C
# =============================================================================

try:
    from smbus2 import SMBus  # type: ignore
except ImportError:  # pragma: no cover
    from smbus import SMBus  # type: ignore  # noqa: F401

_MPU_ADDR = 0x68
_MPU_PWR_MGMT_1 = 0x6B
_MPU_SMPLRT_DIV = 0x19
_MPU_CONFIG = 0x1A
_MPU_GYRO_CONFIG = 0x1B
_MPU_ACCEL_CONFIG = 0x1C
_MPU_ACCEL_XOUT_H = 0x3B
_MPU_WHO_AM_I = 0x75

_ACCEL_LSB_PER_G = 16384.0   # ±2g range
_GYRO_LSB_PER_DPS = 131.0    # ±250 dps range
_DEG2RAD = np.pi / 180.0


def _to_int16(hi: int, lo: int) -> int:
    val = (hi << 8) | lo
    return val - 65536 if val & 0x8000 else val


class RealIMU(IMUDriver):
    """GY-91 (MPU9250) IMU on Raspberry Pi I2C bus.

    Returns body-frame quantities:
      lin_vel_b   — placeholder, returns zeros (training adds noise on this
                    channel; revisit with leg-odometry estimator if drift hurts)
      ang_vel_b   — gyro in rad/s, bias-subtracted
      proj_g_b    — normalized accelerometer reading (gravity unit vector)

    Axis remap (`axis_perm`, `axis_sign`) maps raw IMU axes → robot body frame
    (x=forward, y=left, z=up). Defaults are identity — verify on the robot
    by tilting and watching `proj_g`:
      - tilt forward  → proj_g[0] should go positive
      - tilt left     → proj_g[1] should go positive
      - upright       → proj_g[2] ≈ +1
    """

    def __init__(
        self,
        i2c_bus: int = 1,
        axis_perm: tuple[int, int, int] = (0, 1, 2),
        axis_sign: tuple[float, float, float] = (1.0, 1.0, 1.0),
        gyro_bias: np.ndarray | None = None,
    ) -> None:
        self._bus = SMBus(i2c_bus)
        self._axis_perm = np.asarray(axis_perm, dtype=np.int32)
        self._axis_sign = np.asarray(axis_sign, dtype=np.float32)
        self._gyro_bias = (np.zeros(3, dtype=np.float32)
                          if gyro_bias is None
                          else np.asarray(gyro_bias, dtype=np.float32))
        self._init_mpu()

    def _init_mpu(self) -> None:
        who = self._bus.read_byte_data(_MPU_ADDR, _MPU_WHO_AM_I)
        print(f"[RealIMU] MPU9250 WHO_AM_I = 0x{who:02X} "
              f"(expected 0x71; 0x70/0x73 also seen on clones)")
        self._bus.write_byte_data(_MPU_ADDR, _MPU_PWR_MGMT_1, 0x00)
        time.sleep(0.05)
        self._bus.write_byte_data(_MPU_ADDR, _MPU_PWR_MGMT_1, 0x01)  # PLL X gyro
        self._bus.write_byte_data(_MPU_ADDR, _MPU_SMPLRT_DIV, 0x00)
        self._bus.write_byte_data(_MPU_ADDR, _MPU_CONFIG, 0x03)       # DLPF 41Hz
        self._bus.write_byte_data(_MPU_ADDR, _MPU_GYRO_CONFIG, 0x00)  # ±250 dps
        self._bus.write_byte_data(_MPU_ADDR, _MPU_ACCEL_CONFIG, 0x00) # ±2g
        time.sleep(0.05)

    def set_gyro_bias(self, bias: np.ndarray) -> None:
        """Called by calibrate_imu_gyro after the 3s static capture."""
        self._gyro_bias = np.asarray(bias, dtype=np.float32).reshape(3)

    def _read_raw(self) -> tuple[np.ndarray, np.ndarray]:
        d = self._bus.read_i2c_block_data(_MPU_ADDR, _MPU_ACCEL_XOUT_H, 14)
        ax = _to_int16(d[0], d[1]) / _ACCEL_LSB_PER_G
        ay = _to_int16(d[2], d[3]) / _ACCEL_LSB_PER_G
        az = _to_int16(d[4], d[5]) / _ACCEL_LSB_PER_G
        # bytes 6..7 are temperature — skip
        gx = _to_int16(d[8],  d[9])  / _GYRO_LSB_PER_DPS
        gy = _to_int16(d[10], d[11]) / _GYRO_LSB_PER_DPS
        gz = _to_int16(d[12], d[13]) / _GYRO_LSB_PER_DPS
        accel_g = np.array([ax, ay, az], dtype=np.float32)
        gyro_dps = np.array([gx, gy, gz], dtype=np.float32)
        return accel_g, gyro_dps

    def _to_body(self, v_raw: np.ndarray) -> np.ndarray:
        return (v_raw[self._axis_perm] * self._axis_sign).astype(np.float32)

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        accel_g_raw, gyro_dps_raw = self._read_raw()

        ang_vel_b = self._to_body(gyro_dps_raw) * _DEG2RAD - self._gyro_bias

        accel_b = self._to_body(accel_g_raw)
        n = float(np.linalg.norm(accel_b))
        if n < 1e-6:
            proj_g_b = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            proj_g_b = (accel_b / n).astype(np.float32)

        # Placeholder: training had Unoise on this channel, so 0 is acceptable.
        # Swap in a state estimator (IMU + leg odometry) later if drift hurts.
        lin_vel_b = np.zeros(3, dtype=np.float32)

        return lin_vel_b, ang_vel_b, proj_g_b

    def close(self) -> None:
        try:
            self._bus.close()
        except Exception:
            pass


# =============================================================================
# Joystick command source — pygame
# =============================================================================

class JoystickCommand(CommandSource):
    """Read [vx, vy, wz] from a USB/Bluetooth gamepad via pygame.

    Default mapping (Xbox-style):
      left stick Y  → vx  (push forward → +x)
      left stick X  → vy  (push left → +y)
      right stick X → wz  (push left → +yaw, i.e. counter-clockwise from above)

    Deadzones, scales, and axis assignment are tunable. If no joystick is
    connected, ``read()`` returns zeros — so the robot stands still.
    """

    def __init__(
        self,
        vx_scale: float = 0.4,
        vy_scale: float = 0.3,
        wz_scale: float = 1.0,
        deadzone: float = 0.10,
        axis_vx: int = 1, axis_vx_invert: bool = True,
        axis_vy: int = 0, axis_vy_invert: bool = True,
        axis_wz: int = 3, axis_wz_invert: bool = True,
        joystick_index: int = 0,
    ) -> None:
        try:
            import pygame  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "JoystickCommand needs pygame: pip install pygame"
            ) from e
        self._pygame = pygame
        pygame.init()
        pygame.joystick.init()

        self._joy = None
        if pygame.joystick.get_count() > joystick_index:
            self._joy = pygame.joystick.Joystick(joystick_index)
            self._joy.init()
            print(f"[Joystick] connected: {self._joy.get_name()} "
                  f"({self._joy.get_numaxes()} axes)")
        else:
            print("[Joystick] no device found — commands will be zero.")

        self._vx_s = float(vx_scale)
        self._vy_s = float(vy_scale)
        self._wz_s = float(wz_scale)
        self._dz = float(deadzone)
        self._a_vx, self._inv_vx = int(axis_vx), bool(axis_vx_invert)
        self._a_vy, self._inv_vy = int(axis_vy), bool(axis_vy_invert)
        self._a_wz, self._inv_wz = int(axis_wz), bool(axis_wz_invert)

    def _axis(self, idx: int, invert: bool) -> float:
        if self._joy is None:
            return 0.0
        v = float(self._joy.get_axis(idx))
        if abs(v) < self._dz:
            return 0.0
        v = (v - np.sign(v) * self._dz) / (1.0 - self._dz)  # rescale past deadzone
        return -v if invert else v

    def read(self) -> np.ndarray:
        self._pygame.event.pump()
        vx = self._axis(self._a_vx, self._inv_vx) * self._vx_s
        vy = self._axis(self._a_vy, self._inv_vy) * self._vy_s
        wz = self._axis(self._a_wz, self._inv_wz) * self._wz_s
        return np.array([vx, vy, wz], dtype=np.float32)

    def close(self) -> None:
        try:
            self._pygame.joystick.quit()
            self._pygame.quit()
        except Exception:
            pass
