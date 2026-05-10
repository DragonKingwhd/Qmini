"""Interactive identifier for a single GO-M8010-6: rotation axis + gear-ratio.

Setup:
    机器人必须**吊起来**（脚不接触地面），周围没有障碍。
    一次只识别一个电机；其它电机断电（自由）。

How to run (一次跑一个电机):
    cd ~/Desktop/Qmini
    python3 sim2real/deploy/tests/identify_motor.py --port /dev/ttyUSB1 --id 0
    python3 sim2real/deploy/tests/identify_motor.py --port /dev/ttyUSB1 --id 1
    python3 sim2real/deploy/tests/identify_motor.py --port /dev/ttyUSB0 --id 1
    python3 sim2real/deploy/tests/identify_motor.py --port /dev/ttyUSB0 --id 2

What it does:
    1. 读当前电机端位置 q0
    2. 平滑 ramp 到 q0 + delta（默认 1 rad 电机端 ≈ 9° 关节端，假设减速比 6.33）
    3. 提示你目测：腿绕哪个轴转了？大约转了多少度？
    4. ramp 回 q0，断电

Outputs:
    • 轴判定（yaw / roll / pitch）
    • 减速比估计：电机端 1 rad / 关节端实测度数
        - 比例 ≈ 6.33 → SDK 读数是电机端
        - 比例 ≈ 1.00 → SDK 读数是关节端
"""

from __future__ import annotations

import argparse
import math
import sys
import time

# unitree_actuator_sdk is a local C-extension at /home/pi/unitree_actuator_sdk/lib.
# Prepend it so the import works without a system-wide install.
_SDK_LIB = "/home/pi/unitree_actuator_sdk/lib"
if _SDK_LIB not in sys.path:
    sys.path.insert(0, _SDK_LIB)

from unitree_actuator_sdk import (  # type: ignore  # noqa: E402
    MotorCmd,
    MotorData,
    MotorMode,
    MotorType,
    SerialPort,
    queryMotorMode,
)
try:
    from unitree_actuator_sdk import queryGearRatio  # type: ignore  # noqa: E402
except ImportError:
    def queryGearRatio(_motor_type) -> float:  # noqa: N802
        # GO-M8010-6 nominal gear ratio (datasheet).
        return 6.33

MOTOR_TYPE = MotorType.GO_M8010_6


def _make_cmd(motor_id: int, q: float, kp: float, kd: float) -> MotorCmd:
    cmd = MotorCmd()
    cmd.motorType = MOTOR_TYPE
    cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = q
    cmd.dq = 0.0
    cmd.tau = 0.0
    cmd.kp = kp
    cmd.kd = kd
    return cmd


def _make_data() -> MotorData:
    data = MotorData()
    data.motorType = MOTOR_TYPE
    return data


def read_q(serial: SerialPort, motor_id: int) -> float:
    cmd = _make_cmd(motor_id, 0.0, 0.0, 0.0)
    data = _make_data()
    serial.sendRecv(cmd, data)
    return float(data.q)


def ramp(serial: SerialPort, motor_id: int, q_from: float, q_to: float,
         duration_s: float, kp: float, kd: float, hz: float = 200.0) -> None:
    n = max(1, int(duration_s * hz))
    dt = 1.0 / hz
    data = _make_data()
    for i in range(n + 1):
        s = i / n
        q = q_from + (q_to - q_from) * s
        serial.sendRecv(_make_cmd(motor_id, q, kp, kd), data)
        time.sleep(dt)


def hold(serial: SerialPort, motor_id: int, q: float,
         duration_s: float, kp: float, kd: float, hz: float = 200.0) -> None:
    n = max(1, int(duration_s * hz))
    dt = 1.0 / hz
    data = _make_data()
    for _ in range(n):
        serial.sendRecv(_make_cmd(motor_id, q, kp, kd), data)
        time.sleep(dt)


def disable(serial: SerialPort, motor_id: int) -> None:
    """Send zero-gain command to drop torque."""
    data = _make_data()
    serial.sendRecv(_make_cmd(motor_id, 0.0, 0.0, 0.0), data)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True, help="e.g. /dev/ttyUSB1")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--delta", type=float, default=1.0,
                   help="motor-side delta in rad (default 1.0)")
    p.add_argument("--kp", type=float, default=0.20)
    p.add_argument("--kd", type=float, default=1.0)
    p.add_argument("--ramp", type=float, default=1.5, help="ramp time (s)")
    p.add_argument("--hold", type=float, default=2.0, help="hold time (s)")
    args = p.parse_args()

    print("=" * 60)
    print(f"  电机识别: {args.port}  ID={args.id}")
    print(f"  电机端 delta = {args.delta:.3f} rad "
          f"({math.degrees(args.delta):.1f}°电机端)")
    print(f"  kp={args.kp}  kd={args.kd}")
    print("=" * 60)
    print("⚠️  确认机器人已吊起、脚不触地、周围无人。")
    input("按 Enter 开始（Ctrl+C 中止）...")

    serial = SerialPort(args.port)
    gear = float(queryGearRatio(MOTOR_TYPE))
    print(f"\nSDK gear ratio = {gear:.4f}")

    q0 = read_q(serial, args.id)
    print(f"初始位置 q0 (SDK 读数) = {q0:+.4f} rad")

    try:
        print(f"\n→ 缓慢正转到 q0 + {args.delta:.2f} ...")
        ramp(serial, args.id, q0, q0 + args.delta, args.ramp, args.kp, args.kd)
        hold(serial, args.id, q0 + args.delta, args.hold, args.kp, args.kd)

        q_held = read_q(serial, args.id)
        print(f"  到位后读数 = {q_held:+.4f} rad (Δ_sdk = {q_held - q0:+.4f})")

        print("\n👀 请仔细观察这条腿/关节是如何运动的：")
        print("   (a) yaw   — 腿在水平面内向外/向内转（像八字开合）")
        print("   (b) roll  — 腿在前额面内侧倾（向身体外/内倾倒）")
        print("   (c) pitch — 腿前后摆动（前踢/后摆）")
        axis = ""
        while axis not in ("a", "b", "c"):
            axis = input("输入 a / b / c： ").strip().lower()

        deg_str = input("目测关节大约转了多少度（绝对值，如 9 或 60）： ").strip()
        deg_joint = abs(float(deg_str)) if deg_str else float("nan")

        print("\n→ 缓慢回到 q0 ...")
        ramp(serial, args.id, q0 + args.delta, q0, args.ramp, args.kp, args.kd)
        hold(serial, args.id, q0, 0.5, args.kp, args.kd)
    except KeyboardInterrupt:
        print("\n[中断] 紧急回零...")
        try:
            ramp(serial, args.id, read_q(serial, args.id), q0, 1.0, args.kp, args.kd)
        except Exception:
            pass
        raise
    finally:
        disable(serial, args.id)
        print("[已断电]")

    motor_deg = math.degrees(args.delta)
    axis_name = {"a": "yaw", "b": "roll", "c": "pitch"}[axis]
    print("\n" + "=" * 60)
    print(f"轴判定:        {axis_name}")
    print(f"电机端运动:    {motor_deg:.1f}° = {args.delta:.3f} rad")
    if not math.isnan(deg_joint) and deg_joint > 0.1:
        ratio = motor_deg / deg_joint
        print(f"目测关节运动:  {deg_joint:.1f}°")
        print(f"motor/joint ≈  {ratio:.2f}")
        if abs(ratio - gear) < 1.5:
            print(f"  → SDK 读数是【电机端】（接近减速比 {gear:.2f}）")
            print(f"  → 关节角 = SDK_q / {gear:.2f}")
        elif abs(ratio - 1.0) < 0.5:
            print("  → SDK 读数是【关节端】（比例≈1）")
        else:
            print("  → 比例不典型，重测一次或检查目测精度")
    print("=" * 60)


if __name__ == "__main__":
    main()
