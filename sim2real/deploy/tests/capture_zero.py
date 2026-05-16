"""Re-capture joints.motor_zero_rad — OFFICIAL mechanical-limit method.

Per Qmini_DIY.pdf §3 机器人零位标记: put the robot in the folded pose of
figs 3.1–3.3 and rotate EACH joint hard against its MECHANICAL LIMIT (hold it
there by hand/tool), then record motor q at that limit. At the limit the joint
angle is a KNOWN constant = LIMIT_POSE[i] (the URDF/official act_pos extreme),
so the deploy zero is back-computed:

    motor_zero[i] = q_limit[i] - sign[i] * LIMIT_POSE[i] * GEAR_RATIO

(NOT motor_zero = q — that only holds at joint_q = 0, which is NOT the official
reference. Earlier "straight legs" captures were wrong for exactly this reason.)

Workflow: menu of 10 joints → pick one → push THAT joint hard to its mechanical
limit (folded pose) and hold → Enter → averages q (zero torque) → computes &
writes motor_zero to calibration.yaml. Repeat any joint, any order.

SAFETY: ONLY ever sends kp=kd=0 (zero torque). Never drives the motors.
Robot SUSPENDED. Hold the joint firmly against its hard stop while sampling.

Run:
    cd ~/Desktop/Qmini
    python3 sim2real/deploy/tests/capture_zero.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

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

# 中文部位说明,和 manual_motor_control 一致,方便对着实物分清左右腿
JOINT_ZH = [
    "左腿·髋偏航(绕竖轴/内外旋)",
    "左腿·髋横滚(腿内外侧倾)",
    "左腿·髋俯仰(大腿前后抬)",
    "左腿·膝(小腿前后摆)",
    "左腿·踝(脚掌背屈/跖屈)",
    "右腿·髋偏航(绕竖轴/内外旋)",
    "右腿·髋横滚(腿内外侧倾)",
    "右腿·髋俯仰(大腿前后抬)",
    "右腿·膝(小腿前后摆)",
    "右腿·踝(脚掌背屈/跖屈)",
]

# (port, motor_id) per JOINT_NAMES entry — same mapping as real.py.
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

GEAR_RATIO = 6.33

# 折叠姿态(图 3.1–3.3)下各关节顶到的机械限位角(rad),JOINT_NAMES 顺序。
# 取自官方 RoboTamerSdk4Qmini config.yaml 的 act_pos 极值(折叠方向)。
# pitch 链(hip_pitch/knee/ankle)方向明确;hip_yaw/hip_roll 方向为推测,
# 若标完 --debug 该关节读数不在物理范围内,把对应正负号翻一下重标。
LIMIT_POSE = [
    +0.7,   # hip_yaw_l    [-0.1, 0.7]
    +0.6,   # hip_roll_l   [-0.3, 0.6]
    -2.1,   # hip_pitch_l  [-2.1, 0.0]
    +2.1,   # knee_pitch_l [ 0.0, 2.1]
    -2.5,   # ankle_pitch_l[-2.5, 0.0]
    -0.7,   # hip_yaw_r    [-0.7, 0.1]
    -0.6,   # hip_roll_r   [-0.6, 0.3]
    +2.1,   # hip_pitch_r  [ 0.0, 2.1]
    -2.1,   # knee_pitch_r [-2.1, 0.0]
    +2.5,   # ankle_pitch_r[ 0.0, 2.5]
]

N_SAMPLES = 40
WAKE_ITERS = 10


def load_signs() -> List[float]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    s = (cfg.get("joints", {}) or {}).get("sign")
    if isinstance(s, list) and len(s) == len(JOINT_NAMES):
        return [float(v) for v in s]
    return [1.0] * len(JOINT_NAMES)


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


def load_cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text()) or {}


def current_zeros(cfg: dict) -> List[Optional[float]]:
    z = (cfg.get("joints", {}) or {}).get("motor_zero_rad")
    if isinstance(z, list) and len(z) == len(JOINT_NAMES):
        return [float(v) for v in z]
    return [None] * len(JOINT_NAMES)


def write_zeros(zeros: List[Optional[float]]) -> None:
    cfg = load_cfg()
    old = current_zeros(cfg)
    merged = [zeros[i] if zeros[i] is not None else old[i]
              for i in range(len(JOINT_NAMES))]
    if any(v is None for v in merged):
        merged = [0.0 if v is None else v for v in merged]
    cfg.setdefault("joints", {})["motor_zero_rad"] = [float(v) for v in merged]
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def capture_one(serials, idx: int, sign: float) -> Optional[float]:
    port, mid = JOINT_PORTS[idx]
    name = JOINT_NAMES[idx]
    lim = LIMIT_POSE[idx]
    print(f"\n>>> 把 [{name}] ({JOINT_ZH[idx]}) 按官方折叠姿态(图3.1–3.3)"
          f"\n    顶死到它的机械限位(关节角≈{lim:+.2f} rad),用手/工具抵住别动")
    input("    抵住后按 Enter 开始采集(Ctrl+C 取消本次)...")
    print("    唤醒(零力矩)...")
    for _ in range(WAKE_ITERS):
        try:
            read_q(serials[port], mid)
        except Exception:
            pass
        time.sleep(0.01)
    print(f"    采集 {N_SAMPLES} 次取平均,顶住别松...")
    s: List[float] = []
    for _ in range(N_SAMPLES):
        try:
            q = read_q(serials[port], mid)
        except Exception:
            q = float("nan")
        if q_ok(q):
            s.append(q)
        time.sleep(0.01)
    if len(s) < N_SAMPLES * 0.5:
        print(f"    ❌ 读不到回包({len(s)}/{N_SAMPLES});查供电/接线/ID。未保存。")
        return None
    arr = np.asarray(s, dtype=np.float64)
    q_lim = float(np.mean(arr))
    spread = float(np.max(arr) - np.min(arr))
    # 限位法反算: motor_zero = q_limit - sign * LIMIT_POSE * GEAR
    motor_zero = round(q_lim - sign * lim * GEAR_RATIO, 4)
    tag = "  ⚠️ 抖动大,没顶稳?" if spread > 0.05 else ""
    print(f"    {name}: q@限位={q_lim:+.4f}  sign={sign:+.0f}  "
          f"limit={lim:+.2f}  →  motor_zero={motor_zero:+.4f}  (抖动{spread:.4f}){tag}")
    return motor_zero


def main() -> None:
    print("=" * 60)
    print("  逐关节采集 motor_zero_rad (官方限位法 Qmini_DIY §3)")
    print("=" * 60)
    print("⚠️  机器人【悬空】,摆成图3.1–3.3折叠姿态。全程零力矩,不驱动电机。")
    print("⚠️  采每个关节时:把该关节顶死到机械限位、用力抵住别松。")
    print(f"⚠️  motor_zero = q@限位 - sign·限位角·{GEAR_RATIO};每标完即写 {CONFIG_PATH}")

    ports = sorted({p for p, _ in JOINT_PORTS})
    serials = {p: SerialPort(p) for p in ports}

    signs = load_signs()
    zeros: List[Optional[float]] = [None] * len(JOINT_NAMES)
    base = current_zeros(load_cfg())

    while True:
        print("\n" + "-" * 72)
        for i, n in enumerate(JOINT_NAMES):
            port, mid = JOINT_PORTS[i]
            if zeros[i] is not None:
                st = f"✅本次已采 {zeros[i]:+.4f}"
            elif base[i] is not None:
                st = f"旧值 {base[i]:+.4f}(未重采)"
            else:
                st = "未采"
            print(f"  {i:2d}  {JOINT_ZH[i]:22s} {n:14s} {port} ID={mid}  {st}")
        print("  a=依次采全部   w=写入并退出   q=退出(已写的保留)")
        choice = input("选关节序号 / a / w / q: ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("退出。已写入的保留。")
            return
        if choice == "w":
            write_zeros(zeros)
            print(f"✅ 已写入 {CONFIG_PATH}")
            return
        if choice == "a":
            for i in range(len(JOINT_NAMES)):
                try:
                    z = capture_one(serials, i, signs[i])
                except KeyboardInterrupt:
                    print("\n  跳过该关节")
                    continue
                if z is not None:
                    zeros[i] = z
                    write_zeros(zeros)
                    print("    (已写盘)")
            continue
        try:
            idx = int(choice)
            if not (0 <= idx < len(JOINT_NAMES)):
                raise ValueError
        except ValueError:
            print("  无效输入")
            continue
        try:
            z = capture_one(serials, idx, signs[idx])
        except KeyboardInterrupt:
            print("\n  本次取消")
            continue
        if z is not None:
            zeros[idx] = z
            write_zeros(zeros)
            print(f"    ✅ 已写入 {JOINT_NAMES[idx]} = {z:+.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断]")
