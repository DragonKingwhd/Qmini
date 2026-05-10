"""ONNX policy wrapper + Qmini reference-gait action post-processing.

The policy outputs a 10-D residual in (roughly) [-1, 1]. Mirrors training-side
``QminiReferenceGaitAction.process_actions``:

    target = default_joint_pos + reference_offsets(phase) + raw_action * scale

The reference-gait phase advances every control step; ``BIRLPostProcessor``
takes a ``ReferenceGait`` instance and advances it inside ``step``, exactly
mirroring ``_compute_reference_offsets`` -> ``_advance_phase`` -> sample order
on the training side.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .constants import (
    ACTION_CLIP,
    ACTION_DIM,
    ACTION_SCALE,
    DEFAULT_JOINT_POS_VEC,
    JOINT_LIMIT_HIGH,
    JOINT_LIMIT_LOW,
    NUM_JOINTS,
    OBS_DIM,
)
from .reference_gait import ReferenceGait


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
        """Run one inference. Returns (raw_action[10], elapsed_s)."""
        t0 = time.perf_counter()
        out = self.session.run(
            [self.output_name], {self.input_name: obs.reshape(1, OBS_DIM).astype(np.float32)}
        )[0]
        return out[0].astype(np.float32), time.perf_counter() - t0


class GaitActionPostProcessor:
    """Mirror of QminiReferenceGaitAction on the deploy side."""

    def __init__(self, gait: ReferenceGait):
        self._gait = gait
        self._default = np.asarray(DEFAULT_JOINT_POS_VEC, dtype=np.float32)
        self._lo = np.array(
            [v if v is not None else -np.inf for v in JOINT_LIMIT_LOW], dtype=np.float32
        )
        self._hi = np.array(
            [v if v is not None else +np.inf for v in JOINT_LIMIT_HIGH], dtype=np.float32
        )
        self._last_target = self._default.copy()

    @property
    def last_target(self) -> np.ndarray:
        return self._last_target

    def reset(self) -> None:
        self._last_target = self._default.copy()

    def step(self, raw_action: np.ndarray) -> np.ndarray:
        """One control step.

        Order matches QminiReferenceGaitAction.process_actions():
            1. clip raw action to [-ACTION_CLIP, ACTION_CLIP]
            2. advance the gait phase by one step_dt
            3. compute reference offsets at the *advanced* phase
            4. target = default + offsets + raw * scale, then clamp.
        """
        raw = np.clip(raw_action.astype(np.float32), -ACTION_CLIP, ACTION_CLIP).reshape(NUM_JOINTS)
        self._gait.advance()
        offsets = self._gait.offsets()
        target = self._default + offsets + raw * ACTION_SCALE
        target = np.clip(target, self._lo, self._hi).astype(np.float32)
        self._last_target = target
        return target
