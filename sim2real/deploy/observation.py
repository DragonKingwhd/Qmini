"""Build the 44-D actor observation expected by the trained Qmini policy.

Layout (= ``qmini_env_cfg.py:ObservationsCfg.PolicyCfg``, in declaration order):
    base_lin_vel       (3)   body frame, m/s
    base_ang_vel       (3)   body frame, rad/s
    projected_gravity  (3)   gravity unit vector projected into body frame
    velocity_commands  (3)   [vx_cmd, vy_cmd, wz_cmd]
    joint_pos_rel     (10)   joint_pos - default_joint_pos
    joint_vel_rel     (10)   joint_vel (default_vel = 0 in training)
    last_action       (10)   raw policy output of the previous step
    gait_phase        (2)    [sin(2π·phase), cos(2π·phase)]

No history. ``ObservationsCfg.PolicyCfg`` does not set ``history_length``,
so the policy consumes a single timestep.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    DEFAULT_JOINT_POS_VEC,
    NUM_JOINTS,
    OBS_DIM,
)


class ObservationBuilder:
    def __init__(self) -> None:
        self._default_pos = np.asarray(DEFAULT_JOINT_POS_VEC, dtype=np.float32)
        self._last_action = np.zeros(NUM_JOINTS, dtype=np.float32)

    def reset(self) -> None:
        self._last_action[:] = 0.0

    def set_last_action(self, raw_action: np.ndarray) -> None:
        """Store the policy's raw output to feed back as ``last_action`` next step."""
        self._last_action[:] = np.asarray(raw_action, dtype=np.float32).reshape(NUM_JOINTS)

    def build(
        self,
        *,
        base_lin_vel_b: np.ndarray,        # (3,) m/s, body frame
        base_ang_vel_b: np.ndarray,        # (3,) rad/s, body frame
        projected_gravity_b: np.ndarray,   # (3,) unit-ish, body frame
        velocity_command: np.ndarray,      # (3,) [vx, vy, wz]
        joint_pos: np.ndarray,             # (10,) rad
        joint_vel: np.ndarray,             # (10,) rad/s
        gait_phase_sin_cos: np.ndarray,    # (2,)
    ) -> np.ndarray:
        joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(NUM_JOINTS)
        joint_vel = np.asarray(joint_vel, dtype=np.float32).reshape(NUM_JOINTS)

        obs = np.concatenate([
            np.asarray(base_lin_vel_b, dtype=np.float32).reshape(3),
            np.asarray(base_ang_vel_b, dtype=np.float32).reshape(3),
            np.asarray(projected_gravity_b, dtype=np.float32).reshape(3),
            np.asarray(velocity_command, dtype=np.float32).reshape(3),
            joint_pos - self._default_pos,
            joint_vel,                         # default vel = 0, so rel == abs
            self._last_action,
            np.asarray(gait_phase_sin_cos, dtype=np.float32).reshape(2),
        ]).astype(np.float32)
        assert obs.shape == (OBS_DIM,), f"obs shape mismatch: {obs.shape} expected {OBS_DIM}"
        return obs
