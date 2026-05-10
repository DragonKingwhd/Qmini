"""Mock hardware drivers for testing the control loop without a real robot."""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from ..constants import CONTROL_DT, DEFAULT_JOINT_POS, NUM_JOINTS
from .interfaces import CommandSource, IMUDriver, JointDriver


class StaticIMU(IMUDriver):
    """IMU that returns zero (robot perfectly upright, no motion)."""

    def read(self) -> tuple[float, float, np.ndarray]:
        return 0.0, 0.0, np.zeros(3, dtype=np.float32)


class WigglingIMU(IMUDriver):
    """Sinusoidal pitch/roll oscillation, with a self-consistent gyro signal."""

    def __init__(self, roll_amp: float = 0.05, pitch_amp: float = 0.05, period_s: float = 4.0):
        self.r_amp = roll_amp
        self.p_amp = pitch_amp
        self.omega = 2.0 * math.pi / period_s
        self.t0 = time.perf_counter()

    def read(self) -> tuple[float, float, np.ndarray]:
        t = time.perf_counter() - self.t0
        roll = self.r_amp * math.sin(self.omega * t)
        pitch = self.p_amp * math.cos(self.omega * t)
        # d/dt of euler angles, ignoring small-angle coupling
        roll_rate = self.r_amp * self.omega * math.cos(self.omega * t)
        pitch_rate = -self.p_amp * self.omega * math.sin(self.omega * t)
        gyro = np.array([roll_rate, pitch_rate, 0.0], dtype=np.float32)
        return roll, pitch, gyro


class MockJoints(JointDriver):
    """First-order joint dynamics: state lags target with time constant tau."""

    def __init__(self, tau_s: float = 0.04):
        self.tau = tau_s
        self.pos = np.asarray(DEFAULT_JOINT_POS, dtype=np.float32).copy()
        self.target = self.pos.copy()
        self.t_last = time.perf_counter()
        self.history: list[np.ndarray] = []

    def _step(self) -> None:
        now = time.perf_counter()
        dt = max(0.0, now - self.t_last)
        self.t_last = now
        alpha = 1.0 - math.exp(-dt / max(self.tau, 1e-6))
        prev = self.pos.copy()
        self.pos = self.pos + alpha * (self.target - self.pos)
        self._last_vel = (self.pos - prev) / max(dt, 1e-6)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        self._step()
        if not hasattr(self, "_last_vel"):
            self._last_vel = np.zeros(NUM_JOINTS, dtype=np.float32)
        return self.pos.copy(), self._last_vel.astype(np.float32)

    def send_position(self, target_rad: np.ndarray) -> None:
        t = np.asarray(target_rad, dtype=np.float32).reshape(NUM_JOINTS)
        self.target = t.copy()
        self.history.append(t.copy())
        if len(self.history) > 1000:
            self.history.pop(0)


class ConstantCommand(CommandSource):
    def __init__(self, vx: float = 0.3, wz: float = 0.0):
        self.vx = vx
        self.wz = wz

    def read(self) -> tuple[float, float]:
        return self.vx, self.wz


class WSCommand(CommandSource):
    """Velocity command set externally (e.g. via websocket / keyboard thread)."""

    def __init__(self, vx: float = 0.0, wz: float = 0.0):
        self._lock = threading.Lock()
        self._vx = float(vx)
        self._wz = float(wz)

    def set(self, vx: float | None = None, wz: float | None = None) -> None:
        with self._lock:
            if vx is not None:
                self._vx = float(vx)
            if wz is not None:
                self._wz = float(wz)

    def read(self) -> tuple[float, float]:
        with self._lock:
            return self._vx, self._wz
