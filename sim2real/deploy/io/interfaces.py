"""Abstract hardware driver interfaces for Qmini real-robot deploy.

Concrete drivers (real IMU over serial/I2C, real joint actuators over
CAN/UART) must subclass these and implement the methods. Nothing else in
the deploy package should import vendor-specific libraries — they all
talk to hardware through these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class IMUDriver(ABC):
    """Onboard IMU on the robot's base.

    Returns observation quantities in the *base body frame*:
      - linear velocity (m/s, 3-vector)
      - angular velocity (rad/s, 3-vector)
      - projected gravity (unit vector, 3-vector)

    On a typical setup you do *not* directly measure body-frame linear
    velocity from a 6-axis IMU. Two common solutions:
      1. State estimator (e.g. complementary filter / Kalman with leg odometry)
         that fuses gyro + accel + foot kinematics into a body-vel estimate.
      2. Set ``base_lin_vel`` to zero on deploy and accept a small sim2real
         gap — the policy was trained with noise on this channel.
    Either way, the driver returns a single tuple per call.
    """

    @abstractmethod
    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (base_lin_vel_b[3], base_ang_vel_b[3], projected_gravity_b[3]).

        All in the robot base body frame, float32. Must be non-blocking
        and fast (<1 ms typical).
        """
        ...


class JointDriver(ABC):
    """Bipedal-leg actuator bus (10 joints).

    All vectors are length-10 float arrays, ordered exactly as
    ``constants.JOINT_NAMES``. Concrete drivers must permute internally if
    the hardware-side wiring uses a different order.
    """

    @abstractmethod
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (joint_pos_rad[10], joint_vel_rad_s[10])."""
        ...

    @abstractmethod
    def send_position(self, target_rad: np.ndarray) -> None:
        """Send joint position targets in radians (length-10 array).

        The driver is responsible for any unit conversion (rad -> motor
        encoder counts, degrees, etc.) and for applying per-motor PD gains
        on the firmware side.
        """
        ...

    def emergency_stop(self) -> None:
        """Optional: bring the robot to a safe state. Default: no-op."""
        return None


class CommandSource(ABC):
    """Velocity command input — joystick, websocket, scripted, …"""

    @abstractmethod
    def read(self) -> np.ndarray:
        """Return [vx_cmd, vy_cmd, wz_cmd] (m/s, m/s, rad/s)."""
        ...
