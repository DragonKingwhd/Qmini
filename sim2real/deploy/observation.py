"""Build the 49-D actor observation expected by the trained Qmini policy,
with a 3-step history (final policy input is 147-D).

Layout per step (from mdp/observations.py::PolicyCfg, in order):
    velocity_commands   (2)    [vx_cmd, wz_cmd]
    base_euler          (2)    [roll, pitch]              (rad)
    base_ang_vel        (3)    body-frame ang-vel * 0.5
    joint_pos_rel       (10)   joint_pos - ref_joint_pos
    joint_vel           (10)   joint_vel * 0.1
    joint_pos_err       (10)   current_joint_target - joint_pos
    phase_sig           (4)    [sin_l, sin_r, cos_l, cos_r] * static_flag
    phase_freq          (4)    [(f*0.3-1) x2] * static_flag
    -----------------------
    total = 45

History layer: ``history_length=3`` with ``flatten_history_dim=True`` —
rsl_rl's ObservationManager flattens history first so the resulting
vector is [obs_t-2, obs_t-1, obs_t] (oldest first). On a fresh reset, it
zero-pads the oldest entries.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .constants import (
    ANG_VEL_SCALE,
    JOINT_VEL_SCALE,
    NUM_JOINTS,
    OBS_HISTORY_LEN,
    OBS_PER_STEP,
    REF_JOINT_POS,
    STATIC_CMD_NORM_THRESHOLD,
)


class ObservationBuilder:
    def __init__(self) -> None:
        self._ref = np.asarray(REF_JOINT_POS, dtype=np.float32)
        self._history: deque[np.ndarray] = deque(maxlen=OBS_HISTORY_LEN)
        self.reset()

    def reset(self) -> None:
        self._history.clear()
        for _ in range(OBS_HISTORY_LEN):
            self._history.append(np.zeros(OBS_PER_STEP, dtype=np.float32))

    @staticmethod
    def _static_flag(vx_cmd: float, wz_cmd: float) -> float:
        # Training uses commands[:, :3] which is [vx, vy, wz]; vy is locked to 0.
        cmd_norm = np.linalg.norm([vx_cmd, 0.0, wz_cmd])
        return 1.0 if cmd_norm >= STATIC_CMD_NORM_THRESHOLD else 0.0

    def build(
        self,
        *,
        vx_cmd: float,
        wz_cmd: float,
        roll: float,
        pitch: float,
        ang_vel_xyz: np.ndarray,           # (3,) rad/s, body frame
        joint_pos: np.ndarray,             # (10,) rad
        joint_vel: np.ndarray,             # (10,) rad/s
        current_joint_target: np.ndarray,  # (10,) rad
        pm_phase_sig: np.ndarray,          # (4,)  from PhaseModulator
        pm_freq_sig: np.ndarray,           # (4,)  from PhaseModulator
    ) -> np.ndarray:
        joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(NUM_JOINTS)
        joint_vel = np.asarray(joint_vel, dtype=np.float32).reshape(NUM_JOINTS)
        joint_target = np.asarray(current_joint_target, dtype=np.float32).reshape(NUM_JOINTS)
        ang_vel = np.asarray(ang_vel_xyz, dtype=np.float32).reshape(3)

        s = self._static_flag(vx_cmd, wz_cmd)

        step = np.concatenate([
            np.array([vx_cmd, wz_cmd], dtype=np.float32),         # 2
            np.array([roll, pitch], dtype=np.float32),            # 2
            ang_vel * ANG_VEL_SCALE,                              # 3
            joint_pos - self._ref,                                # 10
            joint_vel * JOINT_VEL_SCALE,                          # 10
            joint_target - joint_pos,                             # 10
            pm_phase_sig.astype(np.float32) * s,                  # 4
            pm_freq_sig.astype(np.float32) * s,                   # 4
        ]).astype(np.float32)
        assert step.shape == (OBS_PER_STEP,), f"per-step obs shape mismatch: {step.shape}"

        self._history.append(step)
        # Oldest first (matches rsl_rl ObservationManager flatten_history_dim=True)
        return np.concatenate(list(self._history)).astype(np.float32)
