"""Run the Qmini control loop against mock drivers and assert basic invariants.

Usage:
    cd /home/user/Desktop/WHD/Qmini/sim2real
    python -m deploy.tests.test_mock_loop \
        --onnx /path/to/exported/policy.onnx \
        --seconds 5
"""

from __future__ import annotations

import argparse

import numpy as np

from ..constants import (
    ACTION_DIM,
    CONTROL_HZ,
    DEFAULT_JOINT_POS,
    JOINT_LIMIT_HIGH,
    JOINT_LIMIT_LOW,
    NUM_JOINTS,
    OBS_DIM,
)
from ..io.mock import ConstantCommand, MockJoints, WigglingIMU
from ..main import QminiController


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--wz", type=float, default=0.0)
    args = ap.parse_args()

    imu = WigglingIMU()
    joints = MockJoints(tau_s=0.04)
    cmd = ConstantCommand(vx=args.vx, wz=args.wz)

    ctrl = QminiController(
        onnx_path=args.onnx,
        imu=imu, joints=joints, cmd_source=cmd,
        record_history=True,
    )
    ctrl.run(duration_s=args.seconds)
    hist = ctrl.history or []
    assert hist, "no steps recorded"

    # 1. action shape and clip
    for r in hist:
        assert r.raw_action.shape == (ACTION_DIM,), f"bad action shape {r.raw_action.shape}"
        assert np.all(np.isfinite(r.raw_action)), "non-finite action"

    # 2. joint targets within configured soft limits (where set)
    lo = np.array([v if v is not None else -np.inf for v in JOINT_LIMIT_LOW])
    hi = np.array([v if v is not None else +np.inf for v in JOINT_LIMIT_HIGH])
    for r in hist:
        assert r.joint_target.shape == (NUM_JOINTS,)
        assert np.all(r.joint_target >= lo - 1e-6), f"target below limit: {r.joint_target}"
        assert np.all(r.joint_target <= hi + 1e-6), f"target above limit: {r.joint_target}"

    # 3. loop rate close to CONTROL_HZ
    total = hist[-1].t - hist[0].t
    rate = (len(hist) - 1) / total if total > 0 else 0
    print(f"  rate: {rate:.1f} Hz (target {CONTROL_HZ:.1f})")
    assert abs(rate - CONTROL_HZ) < CONTROL_HZ * 0.15, f"loop rate off: {rate:.1f}"

    # 4. inference latency
    infs = np.array([r.inference_s for r in hist]) * 1000
    print(f"  inference: mean {infs.mean():.2f} ms, max {infs.max():.2f} ms")

    # 5. frequency outputs in [0.5, 3.5]
    freqs = np.stack([r.freq for r in hist])
    print(f"  phase freq min/max: {freqs.min():.2f} / {freqs.max():.2f}")
    assert freqs.min() >= 0.5 - 1e-3 and freqs.max() <= 3.5 + 1e-3

    # 6. joint targets started at default pose
    diff0 = np.linalg.norm(hist[0].joint_target - np.asarray(DEFAULT_JOINT_POS))
    print(f"  |target_t0 - default_pose|: {diff0:.4f} rad")

    # 7. obs dim sanity
    assert OBS_DIM == 135

    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
