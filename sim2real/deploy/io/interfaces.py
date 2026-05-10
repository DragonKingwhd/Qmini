"""Abstract hardware driver interfaces for Qmini real-robot deploy.

Concrete drivers (real IMU over serial/I2C, real joint actuators over
CAN/UART) must subclass these and implement the methods. Nothing else in
the deploy package should import vendor-specific libraries; they all talk
to hardware through these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class IMUDriver(ABC):
    """Onboard IMU mounted on the robot's base.

    Returns angles in radians and angular velocities in rad/s, in the
    *base body frame*. The training-side observations use roll/pitch from
    a quaternion and angular velocity in body frame, so the real IMU must
    be mounted such that its X axis points forward, Y left, Z up — or the
    driver internally remaps to that convention.
    """

    @abstractmethod
    def read(self) -> tuple[float, float, np.ndarray]:
        """Return (roll_rad, pitch_rad, ang_vel_xyz_rad_s).

        ``ang_vel_xyz`` shape (3,) float32, body frame.
        Must be non-blocking and fast (<1 ms typically).
        """
        ...


class JointDriver(ABC):
    """Bipedal-leg actuator bus (10 joints).

    All vectors are length-10 float arrays, ordered exactly as
    ``constants.JOINT_NAMES``. Implementations must permute internally if
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
        encoder counts, degrees, etc.) and for enforcing per-motor PD
        gains on the firmware side.
        """
        ...

    def emergency_stop(self) -> None:
        """Optional: bring the robot to a safe state. Default: no-op."""
        return None


class CommandSource(ABC):
    """Velocity command input — joystick, WS, scripted, etc.

    Returns the (lin_vel_x, ang_vel_z) command consumed by the policy.
    Lateral velocity is hard-coded to 0 by the training command config.
    """

    @abstractmethod
    def read(self) -> tuple[float, float]:
        """Return (lin_vel_x_m_s, ang_vel_z_rad_s)."""
        ...
