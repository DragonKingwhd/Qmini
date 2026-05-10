"""Mock hardware drivers for testing the control loop without a real robot."""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from ..constants import DEFAULT_JOINT_POS_VEC, NUM_JOINTS
from .interfaces import CommandSource, IMUDriver, JointDriver


class StaticIMU(IMUDriver):
    """IMU that returns zero motion and gravity pointing -z (robot upright)."""

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lin = np.zeros(3, dtype=np.float32)
        ang = np.zeros(3, dtype=np.float32)
        gvec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return lin, ang, gvec


class WigglingIMU(IMUDriver):
    """Mild oscillation, with a self-consistent gyro signal and tilted gravity."""

    def __init__(self, roll_amp: float = 0.05, pitch_amp: float = 0.05, period_s: float = 4.0):
        self.r_amp = roll_amp
        self.p_amp = pitch_amp
        self.omega = 2.0 * math.pi / period_s
        self.t0 = time.perf_counter()

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = time.perf_counter() - self.t0
        roll = self.r_amp * math.sin(self.omega * t)
        pitch = self.p_amp * math.cos(self.omega * t)

        # Project gravity (world -z) into body frame for small angles:
        #   g_body ≈ [sin(pitch), -sin(roll)*cos(pitch), -cos(roll)*cos(pitch)]
        gvec = np.array([
            math.sin(pitch),
            -math.sin(roll) * math.cos(pitch),
            -math.cos(roll) * math.cos(pitch),
        ], dtype=np.float32)

        # body-frame gyro = d/dt of euler (small-angle approx)
        ang = np.array([
            self.r_amp * self.omega * math.cos(self.omega * t),
            -self.p_amp * self.omega * math.sin(self.omega * t),
            0.0,
        ], dtype=np.float32)

        # No actual translation in this mock
        lin = np.zeros(3, dtype=np.float32)
        return lin, ang, gvec


class MockJoints(JointDriver):
    """First-order joint dynamics: state lags target with time constant tau."""

    def __init__(self, tau_s: float = 0.04):
        self.tau = tau_s
        self.pos = np.asarray(DEFAULT_JOINT_POS_VEC, dtype=np.float32).copy()
        self.target = self.pos.copy()
        self.t_last = time.perf_counter()
        self.history: list[np.ndarray] = []
        self._last_vel = np.zeros(NUM_JOINTS, dtype=np.float32)

    def _step(self) -> None:
        now = time.perf_counter()
        dt = max(0.0, now - self.t_last)
        self.t_last = now
        alpha = 1.0 - math.exp(-dt / max(self.tau, 1e-6))
        prev = self.pos.copy()
        self.pos = self.pos + alpha * (self.target - self.pos)
        if dt > 0:
            self._last_vel = ((self.pos - prev) / dt).astype(np.float32)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        self._step()
        return self.pos.copy(), self._last_vel.copy()

    def send_position(self, target_rad: np.ndarray) -> None:
        t = np.asarray(target_rad, dtype=np.float32).reshape(NUM_JOINTS)
        self.target = t.copy()
        self.history.append(t.copy())
        if len(self.history) > 1000:
            self.history.pop(0)


class ConstantCommand(CommandSource):
    def __init__(self, vx: float = 0.10, vy: float = 0.0, wz: float = 0.0):
        self.cmd = np.array([vx, vy, wz], dtype=np.float32)

    def read(self) -> np.ndarray:
        return self.cmd.copy()


class WSCommand(CommandSource):
    """Velocity command set externally (e.g. via websocket / keyboard thread)."""

    def __init__(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0):
        self._lock = threading.Lock()
        self._cmd = np.array([vx, vy, wz], dtype=np.float32)

    def set(self, vx: float | None = None, vy: float | None = None, wz: float | None = None) -> None:
        with self._lock:
            if vx is not None:
                self._cmd[0] = float(vx)
            if vy is not None:
                self._cmd[1] = float(vy)
            if wz is not None:
                self._cmd[2] = float(wz)

    def read(self) -> np.ndarray:
        with self._lock:
            return self._cmd.copy()
