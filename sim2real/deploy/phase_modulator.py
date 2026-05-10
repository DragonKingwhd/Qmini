"""CPU/numpy port of the training-side PhaseModulator.

Mirrors source/Qmini/Qmini/tasks/manager_based/qmini/mdp/actions.py::PhaseModulator
but for a single robot (no batched envs) and using numpy.

The deploy-side phase MUST be advanced with the same dt the training side
used (env.step_dt = decimation * sim.dt = 15 * 0.001 = 0.015 s) and updated
with the *scaled* network frequency output (after [-1,1] -> [low,high] map).
"""

from __future__ import annotations

import math

import numpy as np

from .constants import CONVERT_PHI, NUM_LEGS


class PhaseModulator:
    def __init__(self, time_step: float, rng: np.random.Generator | None = None):
        self._dt = float(time_step)
        rng = rng or np.random.default_rng()
        self._phase = (rng.random(NUM_LEGS) * 2.0 * math.pi).astype(np.float32)
        self._frequency = np.full(NUM_LEGS, 0.5, dtype=np.float32)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng()
        self._phase[:] = (rng.random(NUM_LEGS) * 2.0 * math.pi).astype(np.float32)
        self._frequency[:] = 0.5

    def compute(self, frequency: np.ndarray) -> np.ndarray:
        """Advance phase one step. ``frequency`` shape (2,)."""
        self._frequency[:] = frequency.astype(np.float32)
        self._phase[:] = (self._phase + 2.0 * math.pi * self._frequency * self._dt) % (2.0 * math.pi)
        return self._phase

    @property
    def phase(self) -> np.ndarray:
        return self._phase

    @property
    def frequency(self) -> np.ndarray:
        return self._frequency

    @property
    def pm_phase(self) -> np.ndarray:
        """Concatenated [sin(phase_l), sin(phase_r), cos(phase_l), cos(phase_r)] (4,)."""
        return np.concatenate([np.sin(self._phase), np.cos(self._phase)]).astype(np.float32)

    @property
    def pm_frequency_obs(self) -> np.ndarray:
        """[(f*0.3 - 1) repeated x2] (4,) — matches phase_freq_signal()."""
        x = self._frequency * 0.3 - 1.0
        return np.concatenate([x, x]).astype(np.float32)

    @property
    def support_mask(self) -> np.ndarray:
        return ((self._phase >= 0.0) & (self._phase < CONVERT_PHI)).astype(np.float32)
