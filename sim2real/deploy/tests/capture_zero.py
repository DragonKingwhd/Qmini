"""Re-capture joints.motor_zero_rad at the URDF-zero pose.

motor_zero_rad[i] = motor-side q when joint i is at URDF zero (joint_q = 0):
legs perfectly straight (hips & knees), feet/toes pointing straight DOWN
(ankle neutral). The deploy driver uses it as:
    motor_q = motor_zero + sign * joint_q * GEAR_RATIO
so a wrong/stale motor_zero makes every joint read & command garbage.

These GO-M8010-6 zeros can drift after power cycles / manual handling, so
re-run this whenever joint readings look outside the physical range.

SAFETY: this script ONLY ever sends kp=kd=0 (zero torque). It never drives
the motors. Robot must be SUSPENDED and held by hand at the URDF-zero pose.

Run:
    cd ~/Desktop/Qmini
    python3 sim2real/deploy/tests/capture_zero.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

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

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "sim2real/config/calibration.yaml"
MOTOR_TYPE = MotorType.GO_M8010_6

JOINT_NAMES = [
    "hip_yaw_l", "hip_roll_l", "hip_pitch_l", "knee_pitch_l", "ankle_pitch_l",
    "hip_yaw_r", "hip_roll_r", "hip_pitch_r", "knee_pitch_r", "ankle_pitch_r",
]

# (port, motor_id) per JOINT_NAMES entry — same mapping as real.py / calibrate_sign.
JOINT_PORTS: List[Tuple[str, int]] = [
    ("/dev/ttyUSB0", 1),
    ("/dev/ttyUSB1", 1),
    ("/dev/ttyUSB3", 0),
    ("/dev/ttyUSB3", 1),
    ("/dev/ttyUSB3", 2),
    ("/dev/ttyUSB0", 2),
    ("/dev/ttyUSB1", 0),
    ("/dev/ttyUSB2", 0),
    ("/dev/ttyUSB2", 1),
    ("/dev/ttyUSB2", 2),
]

N_SAMPLES = 40        # averaged for a stable zero
WAKE_ITERS = 10       # zero-torque packets to wake motors before reading


def read_q(serial: SerialPort, motor_id: int) -> float:
    cmd = MotorCmd()
    cmd.motorType = MOTOR_TYPE
    cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = 0.0
    cmd.dq = 0.0
    cmd.tau = 0.0
    cmd.kp = 0.0          # zero torque — never drives the motor
    cmd.kd = 0.0
    data = MotorData()
    data.motorType = MOTOR_TYPE
    serial.sendRecv(cmd, data)
    return float(data.q)


def q_ok(q: float) -> bool:
    return np.isfinite(q) and abs(q) < 1.0e4


def main() -> None:
    print("=" * 64)
    print("  采集 motor_zero_rad  (URDF 零位)")
    print("=" * 64)
    print("⚠️  机器人必须【悬空】。全程零力矩,不会驱动电机。")
    print("⚠️  用手把机器人摆到【URDF 零位】并保持不动:")
    print("      · 两条腿完全【伸直】(髋、膝都不弯)")
    print("      · 脚掌/脚尖【竖直朝下】(踝中位)")
    print("      · 左右对称")
    print(f"⚠️  会覆盖 {CONFIG_PATH} 的 joints.motor_zero_rad")
    input("摆好并扶稳后按 Enter 开始采集...")

    ports = sorted({p for p, _ in JOINT_PORTS})
    serials = {p: SerialPort(p) for p in ports}

    # wake (zero torque) so the first reads aren't timeout garbage
    print("\n唤醒电机(零力矩)...")
    for _ in range(WAKE_ITERS):
        for port, mid in JOINT_PORTS:
            try:
                read_q(serials[port], mid)
            except Exception:
                pass
        time.sleep(0.01)

    print(f"采集 {N_SAMPLES} 次取平均(保持机器人不动)...")
    samples = [[] for _ in range(len(JOINT_NAMES))]
    for _ in range(N_SAMPLES):
        for i, (port, mid) in enumerate(JOINT_PORTS):
            try:
                q = read_q(serials[port], mid)
            except Exception:
                q = float("nan")
            if q_ok(q):
                samples[i].append(q)
        time.sleep(0.01)

    zeros: List[float] = []
    bad: List[str] = []
    print("\n结果 (motor-side rad):")
    for i, name in enumerate(JOINT_NAMES):
        s = samples[i]
        if len(s) < N_SAMPLES * 0.5:
            bad.append(name)
            zeros.append(float("nan"))
            print(f"  {name:16s}  采样不足({len(s)}/{N_SAMPLES}) ⚠️")
            continue
        arr = np.asarray(s, dtype=np.float64)
        z = float(np.mean(arr))
        spread = float(np.max(arr) - np.min(arr))
        zeros.append(round(z, 4))
        warn = "  ⚠️ 抖动大,机器人没扶稳?" if spread > 0.05 else ""
        print(f"  {name:16s}  zero = {z:+.4f}   (抖动 {spread:.4f}){warn}")

    if bad:
        print(f"\n❌ 这些关节读不到回包: {bad};未写入。先查供电/接线再重跑。")
        return

    cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    cfg.setdefault("joints", {})["motor_zero_rad"] = [float(z) for z in zeros]
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"\n✅ 已写入 {CONFIG_PATH}")
    print("下一步: 悬空跑 run_qmini --debug, 看关节读数是否回到物理范围内、能否跟随。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断] 未写入")
