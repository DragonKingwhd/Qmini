"""Real-robot launcher for Qmini on Raspberry Pi.

Default: uses real drivers (UnitreeJointDriver + RealIMU + JoystickCommand).
Pass --mock to fall back to the mock drivers for dry-run on a desktop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from deploy.constants import JOINT_NAMES

_SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve(p: str) -> Path | None:
    """Resolve a path as given, else relative to this script dir (sim2real/).
    Returns None if it cannot be found anywhere — caller MUST handle that
    (running with no calibration silently uses sign=+1/zero=0 = jams robot)."""
    cand = Path(p)
    if cand.exists():
        return cand.resolve()
    alt = _SCRIPT_DIR / p
    if alt.exists():
        return alt.resolve()
    return None
from deploy.main import QminiController


def _print_debug_summary(ctrl) -> None:
    hist = ctrl.history
    if not hist:
        print("[debug] no history recorded")
        return
    tgt = np.stack([h.joint_target for h in hist])   # (T, 10)
    pos = np.stack([h.joint_pos for h in hist])       # (T, 10)
    print("\n" + "=" * 72)
    print("  每关节: 指令幅度 (target range) vs 实测幅度 (measured range)")
    print("  指令大但实测≈0 → 电机/总线没跟上;  指令也≈0 → 策略/步态没让它动")
    print("=" * 72)
    print(f"  {'joint':16s} {'cmd_range':>10s} {'meas_range':>11s} "
          f"{'cmd_min':>8s} {'cmd_max':>8s} {'meas_min':>9s} {'meas_max':>9s}")
    for i, name in enumerate(JOINT_NAMES):
        cr = float(tgt[:, i].max() - tgt[:, i].min())
        mr = float(pos[:, i].max() - pos[:, i].min())
        flag = "  <-- 指令动了但没跟上" if (cr > 0.10 and mr < 0.02) else ""
        print(f"  {name:16s} {cr:10.3f} {mr:11.3f} "
              f"{tgt[:, i].min():8.3f} {tgt[:, i].max():8.3f} "
              f"{pos[:, i].min():9.3f} {pos[:, i].max():9.3f}{flag}")
    print("=" * 72)


def _build_real(args, cfg_path: Path):
    from deploy.io.real import JoystickCommand, RealIMU, UnitreeJointDriver
    imu = RealIMU(i2c_bus=args.i2c_bus)
    joints = UnitreeJointDriver(
        zero_offset_yaml=str(cfg_path) if cfg_path is not None else None,
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
    ap.add_argument("--no-ramp", action="store_true",
                    help="Skip the soft-start ramp to DEFAULT (NOT recommended).")
    ap.add_argument("--ramp-secs", type=float, default=3.0,
                    help="Soft-start ramp duration (s).")
    ap.add_argument("--debug", action="store_true",
                    help="Record history and print per-joint cmd vs measured "
                         "amplitude summary at the end.")
    ap.add_argument("--allow-no-calib", action="store_true",
                    help="DANGEROUS: run even if calibration.yaml not found.")
    args = ap.parse_args()

    onnx_path = _resolve(args.onnx)
    cfg_path = _resolve(args.config)

    if not args.mock:
        if onnx_path is None:
            sys.exit(f"[FATAL] ONNX not found: {args.onnx!r} "
                     f"(tried cwd and {_SCRIPT_DIR})")
        if cfg_path is None and not args.allow_no_calib:
            sys.exit(
                "\n" + "!" * 64 +
                f"\n[FATAL] 标定文件没找到: {args.config!r}"
                f"\n  试过: ./{args.config}  和  {_SCRIPT_DIR / args.config}"
                "\n  没有标定会用 sign=+1/zero=0 → 电机全顶死/乱弹(这正是之前的 bug)。"
                "\n  确认 sim2real/config/calibration.yaml 存在;"
                "\n  非要无标定测试加 --allow-no-calib(危险)。\n" + "!" * 64)
        if cfg_path is not None:
            print(f"[INFO] calibration: {cfg_path}")
        print(f"[INFO] onnx: {onnx_path}")

    imu, joints, cmd = (_build_mock(args) if args.mock
                        else _build_real(args, cfg_path))

    ctrl = QminiController(
        onnx_path=onnx_path if onnx_path is not None else Path(args.onnx),
        imu=imu, joints=joints, cmd_source=cmd,
        calibration_yaml=cfg_path,
        record_history=args.debug,
    )

    print("[INFO] checking initial joint pose...")
    ctrl.check_pose()

    if not args.no_ramp:
        print("[INFO] soft-start: ramping measured pose -> DEFAULT...")
        ctrl.ramp_to_default(duration_s=args.ramp_secs)

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
        if args.debug:
            try:
                _print_debug_summary(ctrl)
            except Exception as e:
                print(f"[debug] summary failed: {e!r}")


if __name__ == "__main__":
    main()
