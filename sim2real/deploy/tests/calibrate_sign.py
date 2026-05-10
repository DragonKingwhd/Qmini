"""Sign calibration: verify each joint moves in URDF-positive direction.

Prereqs:
    1. calibration.yaml has motor_zero_rad (10 values) populated.
    2. Robot is hanging in the URDF zero pose (legs vertical, toes down).
    3. Joints can move freely (no fault state on any motor).

What it does:
    For each of 10 joints in JOINT_NAMES order:
      - sends a small joint-side delta (within URDF limits) at low PD gain
      - asks user: "did the leg move in the described direction?" (y/n/s)
      - records sign = +1 if yes, -1 if no
      - ramps back to motor_zero, drops torque on that joint
    Writes joints.sign back to calibration.yaml.

Run:
    cd ~/Desktop/Qmini
    python3 sim2real/deploy/tests/calibrate_sign.py
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

GEAR_RATIO = 6.33
MOTOR_TYPE = MotorType.GO_M8010_6
KP = 1.5
KD = 1.0
RAMP_S = 1.8
HOLD_S = 1.8
LOOP_HZ = 200.0

JOINT_NAMES = [
    "hip_yaw_l", "hip_roll_l", "hip_pitch_l", "knee_pitch_l", "ankle_pitch_l",
    "hip_yaw_r", "hip_roll_r", "hip_pitch_r", "knee_pitch_r", "ankle_pitch_r",
]

# (port, motor_id) for each JOINT_NAMES entry.
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

# Per-joint test:
#   delta: joint-side rad, picked to stay within URDF limits even if sign is wrong
#   description: physical motion to expect *if sign=+1 is correct*
#                (derived from URDF axes: +Z up, +Y left, +X forward)
JOINT_TESTS = [
    (+0.05, "左脚尖朝身体右侧转 (内旋)"),
    (-0.10, "左腿向身体内侧倾 (左脚朝身体右侧靠)"),
    (-0.15, "左大腿向身体前方抬起"),
    (+0.15, "左小腿向身体前方摆动"),
    (-0.15, "左脚尖向身体前方旋转 (脚背勾向小腿)"),
    (-0.05, "右脚尖朝身体左侧转 (内旋)"),
    (+0.10, "右腿向身体内侧倾 (右脚朝身体左侧靠)"),
    (+0.15, "右大腿向身体前方抬起"),
    (-0.15, "右小腿向身体前方摆动"),
    (+0.15, "右脚尖向身体前方旋转"),
]


def load_motor_zeros(path: Path) -> np.ndarray:
    cfg = yaml.safe_load(path.read_text())
    zeros = cfg["joints"]["motor_zero_rad"]
    arr = np.asarray(zeros, dtype=np.float64)
    if arr.shape != (10,):
        raise ValueError(f"motor_zero_rad must have length 10, got {arr.shape}")
    return arr


def _make_cmd(motor_id: int, motor_q: float, kp: float, kd: float) -> MotorCmd:
    cmd = MotorCmd()
    cmd.motorType = MOTOR_TYPE
    cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = float(motor_q)
    cmd.dq = 0.0
    cmd.tau = 0.0
    cmd.kp = float(kp)
    cmd.kd = float(kd)
    return cmd


def send(serial: SerialPort, cmd: MotorCmd) -> Tuple[float, float]:
    data = MotorData()
    data.motorType = MOTOR_TYPE
    serial.sendRecv(cmd, data)
    return float(data.q), float(data.dq)


def ramp_motor(serial: SerialPort, motor_id: int, q_from: float, q_to: float,
               duration_s: float, kp: float, kd: float) -> float:
    n = max(1, int(duration_s * LOOP_HZ))
    dt = 1.0 / LOOP_HZ
    last_q = q_from
    for i in range(n + 1):
        s = i / n
        q = q_from + (q_to - q_from) * s
        last_q, _ = send(serial, _make_cmd(motor_id, q, kp, kd))
        time.sleep(dt)
    return last_q


def hold_motor(serial: SerialPort, motor_id: int, q_target: float,
               duration_s: float, kp: float, kd: float) -> float:
    n = max(1, int(duration_s * LOOP_HZ))
    dt = 1.0 / LOOP_HZ
    last_q = q_target
    for _ in range(n):
        last_q, _ = send(serial, _make_cmd(motor_id, q_target, kp, kd))
        time.sleep(dt)
    return last_q


def disable(serial: SerialPort, motor_id: int) -> None:
    send(serial, _make_cmd(motor_id, 0.0, 0.0, 0.0))


def main() -> None:
    print("=" * 64)
    print("  关节方向 (sign) 标定")
    print("=" * 64)
    print("⚠️  机器人吊起、当前在 URDF 零位（双腿垂直、对齐、脚尖朝下）。")
    print("⚠️  使用低增益、小幅运动；任何异常立刻 Ctrl+C。")
    print(f"⚠️  会读写 {CONFIG_PATH}")
    input("按 Enter 继续...")

    motor_zeros = load_motor_zeros(CONFIG_PATH)
    print("\nmotor_zero_rad (10 joints):")
    for n, z in zip(JOINT_NAMES, motor_zeros):
        print(f"  {n:18s} = {z:+.3f}")

    ports = sorted({p for p, _ in JOINT_PORTS})
    serials = {p: SerialPort(p) for p in ports}
    print(f"\n已打开总线: {list(serials.keys())}")

    signs: List[float] = [1.0] * len(JOINT_NAMES)

    for i in range(len(JOINT_NAMES)):
        joint = JOINT_NAMES[i]
        port, motor_id = JOINT_PORTS[i]
        delta, desc = JOINT_TESTS[i]
        motor_zero = float(motor_zeros[i])
        motor_target = motor_zero + 1.0 * GEAR_RATIO * delta  # assume sign=+1

        print("\n" + "-" * 64)
        print(f"[{i+1}/10] {joint}  ({port} ID={motor_id})")
        print(f"  joint δ = {delta:+.3f} rad   motor target = {motor_target:+.3f} rad")
        print(f"  预期效果 (若 sign=+1 正确): {desc}")
        input("  按 Enter 开始...")

        q_started_at = motor_zero
        try:
            # read actual current q first
            q_now, _ = send(serials[port], _make_cmd(motor_id, motor_zero, 0.0, 0.0))
            print(f"  当前 q = {q_now:+.3f}, motor_zero = {motor_zero:+.3f}")
            if abs(q_now - motor_zero) > 0.5:
                print(f"  ⚠️  当前位置离零位 > 0.5 rad；可能是不在零位姿态了。")
                ans = input("  仍然继续? (y/N): ").strip().lower()
                if ans != "y":
                    print("  跳过。")
                    continue
            q_started_at = q_now

            print(f"  → ramp 到 {motor_target:+.3f}...")
            q_at_target = ramp_motor(serials[port], motor_id, q_now, motor_target, RAMP_S, KP, KD)
            achieved = abs(q_at_target - motor_target) < abs(motor_target - q_now) * 0.4 + 0.05
            print(f"    到位读数 = {q_at_target:+.3f}  ({'OK' if achieved else 'NOT REACHED'})")
            if not achieved:
                print("  ⚠️  电机没跟到位 (限位/故障/卡住)；这次结果可能不准。")
            hold_motor(serials[port], motor_id, motor_target, HOLD_S, KP, KD)

            print("\n  问: 关节运动方向跟上面描述一致吗?")
            print("    y - 一致     (sign = +1)")
            print("    n - 反向了   (sign = -1)")
            print("    s - 跳过     (保留 +1，稍后手动改)")
            ans = ""
            while ans not in ("y", "n", "s"):
                ans = input("  输入 y/n/s: ").strip().lower()
            if ans == "y":
                signs[i] = +1.0
                print(f"  ✅ {joint}: sign = +1")
            elif ans == "n":
                signs[i] = -1.0
                print(f"  ✅ {joint}: sign = -1 (反)")
            else:
                signs[i] = +1.0
                print(f"  ⏭  {joint}: skipped, kept +1")
        except KeyboardInterrupt:
            print("\n  [中断]")
            break
        finally:
            try:
                print("  ← 回到 motor_zero...")
                q_back_from = q_at_target if "q_at_target" in dir() else motor_target
                ramp_motor(serials[port], motor_id, q_back_from, motor_zero, RAMP_S, KP, KD)
                disable(serials[port], motor_id)
                print("  [已断电]")
            except Exception as e:
                print(f"  [WARN] 退出时出错: {e!r}")

    # write back
    cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    cfg.setdefault("joints", {})["sign"] = [float(s) for s in signs]
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    print("\n" + "=" * 64)
    print(f"已写入 {CONFIG_PATH}\n")
    for joint, s in zip(JOINT_NAMES, signs):
        print(f"  {joint:18s}  sign = {s:+.0f}")
    print("\n下一步建议: 检查上面的 sign 列, 看左/右是否对称合理")
    print("  对称含义: 左/右相同关节的 sign 通常相同（hip_yaw, hip_roll, hip_pitch, knee, ankle 都成对）")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断]")
