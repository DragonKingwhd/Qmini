"""Real-robot launcher for Qmini on Raspberry Pi.

Default: uses real drivers (UnitreeJointDriver + RealIMU + JoystickCommand).
Pass --mock to fall back to the mock drivers for dry-run on a desktop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deploy.main import QminiController


def _build_real(args):
    from deploy.io.real import JoystickCommand, RealIMU, UnitreeJointDriver
    imu = RealIMU(i2c_bus=args.i2c_bus)
    joints = UnitreeJointDriver(
        zero_offset_yaml=args.config if Path(args.config).exists() else None,
    )
    if args.constant_cmd:
        from deploy.io.mock import ConstantCommand
        cmd = ConstantCommand(vx=args.vx, vy=args.vy, wz=args.wz)
    else:
        cmd = JoystickCommand()
    return imu, joints, cmd


def _build_mock(args):
    from deploy.io.mock import ConstantCommand, MockJoints, WigglingIMU
    imu = WigglingIMU()
    joints = MockJoints()
    cmd = ConstantCommand(vx=args.vx, vy=args.vy, wz=args.wz)
    return imu, joints, cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="policy.onnx")
    ap.add_argument("--config", default="config/calibration.yaml")
    ap.add_argument("--mock", action="store_true",
                    help="Use mock drivers (desktop dry-run).")
    ap.add_argument("--constant-cmd", action="store_true",
                    help="Use --vx/--vy/--wz constant velocity instead of joystick.")
    ap.add_argument("--i2c-bus", type=int, default=1)
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None,
                    help="Stop after N seconds (default: run until Ctrl+C)")
    ap.add_argument("--skip-imu-calib", action="store_true")
    args = ap.parse_args()

    imu, joints, cmd = _build_mock(args) if args.mock else _build_real(args)

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
    finally:
        for obj in (imu, cmd):
            close = getattr(obj, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    main()
