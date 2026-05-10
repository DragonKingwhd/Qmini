"""Numpy CPU port of QminiReferenceGaitAction.

Reproduces the reference walking offsets that are added on top of the
default joint pose before the policy's residual is applied. Faithful
mirror of source/.../mdp/actions.py::QminiReferenceGaitAction:

    target = default_joint_pos + reference_offsets(phase) + raw_action * scale

State carried between steps:
- ``phase`` in [0, 1), advanced each step by step_dt / gait_period.

The right leg's pitch joints use mirrored signs in the Qmini default pose,
so their offsets get a leading minus.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import (
    ANKLE_PITCH_AMPLITUDE,
    CONTROL_DT,
    GAIT_PERIOD_S,
    GAIT_STANCE_RATIO,
    HIP_PITCH_AMPLITUDE,
    JOINT_NAMES,
    KNEE_PITCH_AMPLITUDE,
    NUM_JOINTS,
    PUSH_OFF_ANKLE_SCALE,
)


def _joint_index(name: str) -> int | None:
    return JOINT_NAMES.index(name) if name in JOINT_NAMES else None


_LEFT_HIP   = _joint_index("hip_pitch_l")
_RIGHT_HIP  = _joint_index("hip_pitch_r")
_LEFT_KNEE  = _joint_index("knee_pitch_l")
_RIGHT_KNEE = _joint_index("knee_pitch_r")
_LEFT_ANKLE = _joint_index("ankle_pitch_l")
_RIGHT_ANKLE= _joint_index("ankle_pitch_r")


def _gait_profile(phase: float) -> tuple[float, float, float]:
    """Return (hip, knee, ankle) reference signal at a single phase point."""
    sr = GAIT_STANCE_RATIO
    if phase < sr:
        # ---- stance ----
        progress = phase / sr
        hip = 1.0 - 2.0 * progress
        knee = 0.15 * math.sin(math.pi * progress)
        push_off = max(0.0, (progress - 0.60) / 0.40)
        ankle = -0.55 * hip + PUSH_OFF_ANKLE_SCALE * push_off
    else:
        # ---- swing ----
        progress = (phase - sr) / (1.0 - sr)
        hip = -1.0 + 2.0 * progress
        knee = math.sin(math.pi * progress)
        ankle = -0.35 * hip - 0.20 * math.sin(math.pi * progress)
    return hip, knee, ankle


class ReferenceGait:
    def __init__(self, gait_period_s: float = GAIT_PERIOD_S, step_dt: float = CONTROL_DT,
                 init_phase: float = 0.0):
        self._period = max(gait_period_s, step_dt)
        self._dt = step_dt
        self._phase = float(init_phase) % 1.0

    def reset(self, phase: float | None = None, rng: np.random.Generator | None = None) -> None:
        if phase is not None:
            self._phase = float(phase) % 1.0
        elif rng is not None:
            self._phase = float(rng.random())
        else:
            self._phase = 0.0

    def advance(self) -> float:
        """Advance phase by one control step. Returns the new phase in [0, 1)."""
        self._phase = (self._phase + self._dt / self._period) % 1.0
        return self._phase

    @property
    def phase(self) -> float:
        return self._phase

    @property
    def phase_obs(self) -> np.ndarray:
        """[sin(2π·phase), cos(2π·phase)] — matches mdp/observations.py:gait_phase_obs."""
        a = 2.0 * math.pi * self._phase
        return np.array([math.sin(a), math.cos(a)], dtype=np.float32)

    def offsets(self) -> np.ndarray:
        """Per-joint reference offsets to add on top of the default pose. Shape (10,).

        IMPORTANT: this does *not* advance the phase; call ``advance()`` first
        (or after, but consistently) to match training. The training side
        advances inside ``_compute_reference_offsets`` *before* sampling, so
        we do the same in the controller wrapper.
        """
        offsets = np.zeros(NUM_JOINTS, dtype=np.float32)
        left_phase = self._phase
        right_phase = (self._phase + 0.5) % 1.0
        lh, lk, la = _gait_profile(left_phase)
        rh, rk, ra = _gait_profile(right_phase)

        if _LEFT_HIP    is not None: offsets[_LEFT_HIP]    =  HIP_PITCH_AMPLITUDE   * lh
        if _RIGHT_HIP   is not None: offsets[_RIGHT_HIP]   = -HIP_PITCH_AMPLITUDE   * rh
        if _LEFT_KNEE   is not None: offsets[_LEFT_KNEE]   =  KNEE_PITCH_AMPLITUDE  * lk
        if _RIGHT_KNEE  is not None: offsets[_RIGHT_KNEE]  = -KNEE_PITCH_AMPLITUDE  * rk
        if _LEFT_ANKLE  is not None: offsets[_LEFT_ANKLE]  =  ANKLE_PITCH_AMPLITUDE * la
        if _RIGHT_ANKLE is not None: offsets[_RIGHT_ANKLE] = -ANKLE_PITCH_AMPLITUDE * ra
        return offsets