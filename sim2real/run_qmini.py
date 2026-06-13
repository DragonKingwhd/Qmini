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
    imu = RealIMU(i2c_bus=args.i2c_bus,
                  axis_perm=tuple(args.imu_perm),
                  axis_sign=tuple(args.imu_sign))
    joints = UnitreeJointDriver(
        zero_offset_yaml=str(cfg_path) if cfg_path is not None else None,
        bus_gap_s=args.bus_gap,
        kp_scale=args.kp_scale,
    )
    if args.keyboard:
        from deploy.io.keyboard import KeyboardCommand
        cmd = KeyboardCommand()
    elif args.constant_cmd:
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
    ap.add_argument("--keyboard", action="store_true",
                    help="键盘速度控制: w/s=vx±0.05(0~0.3) a/d=wz±0.1(±0.5) 空格=归零。")
    ap.add_argument("--no-wait", action="store_true",
                    help="跳过'保持 DEFAULT 等回车'的确认门,IMU 标定后直接进策略。")
    ap.add_argument("--i2c-bus", type=int, default=1)
    ap.add_argument("--imu-perm", type=int, nargs=3, default=[0, 1, 2],
                    help="IMU 轴重排到 body(x前y左z上)。先用 debug/check_imu_frame.py 验。")
    ap.add_argument("--imu-sign", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                    help="IMU 各 body 轴乘 ±1(配合 --imu-perm)。")
    ap.add_argument("--bus-gap", type=float, default=0.002,
                    help="485 per-motor turnaround gap (s). 加了 120Ω 终端电阻后"
                         "可试 0.0006 换余量。")
    ap.add_argument("--kp-scale", type=float, default=1.0,
                    help="PD 增益缩放。诊断用:0.4 干净而 1.0 丢包 → 负载电流"
                         "电气干扰实锤(策略动力学会变,正式跑用 1.0)。")
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
    ap.add_argument("--force-pose", action="store_true",
                    help="DANGEROUS: continue even if initial pose check fails "
                         "(≈1.0 rad/joint 偏差=电机重启过,零位作废,会怼硬限位过流).")
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
    bad = ctrl.check_pose()
    if bad and not args.mock and not args.force_pose:
        sys.exit(
            "\n" + "!" * 64 +
            f"\n[FATAL] 关节 {bad} 初始读数偏离 DEFAULT 超过 0.15 rad,拒绝启动。"
            "\n  偏差 ≈ ±1.0 rad 的整数倍 → 那个电机断电/重启过,零位已作废;"
            "\n  继续 ramp 会把真实关节怼向硬限位 → 堵转过流 merror=5 快闪。"
            "\n  处理: 电机电池断电5s重上电(清故障) → 摆台架直腿 →"
            "\n        python3 sim2real/calib/stand_zero.py → goto_default 验证。"
            "\n  确认姿态没问题非要跑: 加 --force-pose(危险)。\n" + "!" * 64)

    if not args.no_ramp:
        print("[INFO] soft-start: ramping measured pose -> DEFAULT...")
        ctrl.ramp_to_default(duration_s=args.ramp_secs)

    if not args.skip_imu_calib:
        print("[INFO] hold robot still for IMU gyro bias calibration (3s)...")
        ctrl.calibrate_imu(duration_s=3.0)

    if not args.mock and not args.no_wait:
        print("\n" + "=" * 60)
        print("  电机正以全 PD 刚度保持 DEFAULT(板载闭环,等待期间一直锁住)。")
        print("  现在可以把机器人从台架拿下来放到地面、扶稳。")
        if args.keyboard:
            print("  键盘: w/s=加减速(0~0.3)  a/d=左/右转(±0.5)  空格=原地踏步")
        print("=" * 60)
        input(">>> 摆好扶稳后按回车,策略接管 <<< ")

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
