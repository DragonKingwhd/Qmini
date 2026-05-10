"""Real-robot launcher for Qmini on Raspberry Pi.

Edit the imports below to point at your `deploy.io.real` driver classes
once you have written them. Until then, this script falls back to mock
drivers so you can verify the pipeline end-to-end on the Pi.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deploy.io.mock import ConstantCommand, MockJoints, WigglingIMU
from deploy.main import QminiController

# When you have a real driver, replace the import line above with:
#   from deploy.io.real import RealIMU, RealJoints, JoystickCommand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="policy.onnx")
    ap.add_argument("--config", default="config/calibration.yaml")
    ap.add_argument("--vx", type=float, default=0.10)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None,
                    help="Stop after N seconds (default: run until Ctrl+C)")
    ap.add_argument("--skip-imu-calib", action="store_true")
    args = ap.parse_args()

    imu = WigglingIMU()              # TODO: RealIMU(...)
    joints = MockJoints()            # TODO: RealJoints(...)
    cmd = ConstantCommand(vx=args.vx, vy=args.vy, wz=args.wz)  # TODO: JoystickCommand(...)

    ctrl = QminiController(
        onnx_path=Path(args.onnx),
        imu=imu, joints=joints, cmd_source=cmd,
        calibration_yaml=Path(args.config) if Path(args.config).exists() else None,
        record_history=False,
    )

    print("[INFO] checking initial joint pose...")
    ctrl.check_pose()

    if not args.skip_imu_calib:
        print("[INFO] hold robot still for IMU gyro bias calibration (3s)...")
        ctrl.calibrate_imu(duration_s=3.0)

    print("[INFO] starting control loop. Ctrl+C to stop.")
    try:
        ctrl.run(duration_s=args.duration)
    except KeyboardInterrupt:
        ctrl.stop()


if __name__ == "__main__":
    main()
