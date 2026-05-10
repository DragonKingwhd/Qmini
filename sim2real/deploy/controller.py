"""ONNX policy wrapper + BIRL action post-processing.

The policy's raw output (12,) is in [-1, 1]. Mirrors training-side
BIRLActionTerm.process_actions:

    raw  -> clip([-1, 1]) -> scale to [INC_LOW, INC_HIGH]
    freq           = scaled[0:2]              -> drives the PhaseModulator
    joint_deltas   = scaled[2:12] (rad/s)     -> integrated:
                       current_joint_target += joint_deltas * step_dt
                     then clipped to joint soft limits.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .constants import (
    ACTION_CLIP,
    ACTION_DIM,
    CONTROL_DT,
    DEFAULT_JOINT_POS,
    INC_HIGH,
    INC_LOW,
    JOINT_LIMIT_HIGH,
    JOINT_LIMIT_LOW,
    NUM_JOINTS,
    NUM_LEGS,
    OBS_DIM,
)


class ONNXPolicy:
    def __init__(self, onnx_path: str | Path):
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        in_shape = self.session.get_inputs()[0].shape
        out_shape = self.session.get_outputs()[0].shape
        if in_shape[-1] != OBS_DIM or out_shape[-1] != ACTION_DIM:
            raise ValueError(
                f"ONNX shape mismatch: expected in={OBS_DIM}, out={ACTION_DIM}; "
                f"got in={in_shape}, out={out_shape}"
            )

    def infer(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        """Run one inference. Returns (raw_action[12], elapsed_s)."""
        t0 = time.perf_counter()
        out = self.session.run(
            [self.output_name], {self.input_name: obs.reshape(1, OBS_DIM).astype(np.float32)}
        )[0]
        return out[0].astype(np.float32), time.perf_counter() - t0


class BIRLPostProcessor:
    """Mirrors BIRLActionTerm.process_actions on the deploy side.

    Holds the integrated joint target between steps. Reset before every run.
    """

    def __init__(self) -> None:
        self._inc_low = np.asarray(INC_LOW, dtype=np.float32)
        self._inc_high = np.asarray(INC_HIGH, dtype=np.float32)
        # Build per-joint hard clamp from JOINT_LIMIT_*; ±inf where unset.
        self._lo = np.array(
            [v if v is not None else -np.inf for v in JOINT_LIMIT_LOW], dtype=np.float32
        )
        self._hi = np.array(
            [v if v is not None else +np.inf for v in JOINT_LIMIT_HIGH], dtype=np.float32
        )
        self._target = np.asarray(DEFAULT_JOINT_POS, dtype=np.float32).copy()

    def reset(self) -> None:
        self._target = np.asarray(DEFAULT_JOINT_POS, dtype=np.float32).copy()

    @property
    def current_joint_target(self) -> np.ndarray:
        return self._target

    def step(self, raw_action: np.ndarray, step_dt: float = CONTROL_DT) -> tuple[np.ndarray, np.ndarray]:
        """Process raw action.

        Returns (frequency[2], joint_target[10]).
        """
        raw = np.clip(raw_action.astype(np.float32), -ACTION_CLIP, ACTION_CLIP)
        scaled = 0.5 * (raw + 1.0) * (self._inc_high - self._inc_low) + self._inc_low

        freq = scaled[:NUM_LEGS]
        joint_deltas = scaled[NUM_LEGS:NUM_LEGS + NUM_JOINTS]

        self._target = self._target + joint_deltas * step_dt
        self._target = np.clip(self._target, self._lo, self._hi).astype(np.float32)
        return freq, self._target
