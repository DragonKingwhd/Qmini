"""只补 hip_roll_l / hip_roll_r 两个关节的 motor_zero,其它 8 个关节一律不动。

为什么单独补:hip_roll 行程小、没有可靠机械硬限位,双限位法(capture_zero 的 a)
解出的 zero 是垃圾(齿比 18~20)。但 hip_roll 的 URDF <origin rpy> = 0,所以它的
joint=0 就是"腿竖直、左右平行、不向内外侧倾"这个姿态 —— 这个姿态你能轻松摆。
在 joint=0 处 motor_zero = 当前 motor_q(与 sign / 齿比 / 限位都无关,最干净)。

安全:全程只发 kp=kd=0(零力矩),绝不驱动电机。

用法(机器人悬空或落地都行,关键是两腿摆竖直平行):
    cd ~/Desktop/Qmini
    python3 sim2real/calib/fix_hip_roll_zero.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

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

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "sim2real/config/calibration.yaml"
MOTOR_TYPE = MotorType.GO_M8010_6

# hip_roll 都在 /dev/ttyUSB1 上(与 real.py / capture_zero.py 一致)。
# motor_zero_rad 列表索引(= JOINT_NAMES 顺序): hip_roll_l=1, hip_roll_r=6。
PORT = "/dev/ttyUSB1"
HIP_ROLL = [
    # (名称, motor_id, 在 motor_zero_rad 列表里的索引)
    ("hip_roll_l", 1, 1),
    ("hip_roll_r", 0, 6),
]

N_SAMPLES = 30
WAKE_ITERS = 10
BUS_GAP = 0.002


def read_q(serial: SerialPort, motor_id: int) -> float:
    cmd = MotorCmd()
    cmd.motorType = MOTOR_TYPE
    cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = 0.0
    cmd.dq = 0.0
    cmd.tau = 0.0
    cmd.kp = 0.0          # 零力矩,绝不驱动电机
    cmd.kd = 0.0
    data = MotorData()
    data.motorType = MOTOR_TYPE
    serial.sendRecv(cmd, data)
    return float(data.q)


def sample(serial: SerialPort, motor_id: int) -> float:
    for _ in range(WAKE_ITERS):
        try:
            read_q(serial, motor_id)
        except Exception:
            pass
        time.sleep(BUS_GAP)
    s: List[float] = []
    for _ in range(N_SAMPLES):
        try:
            q = read_q(serial, motor_id)
        except Exception:
            q = float("nan")
        if np.isfinite(q) and abs(q) < 1.0e4:
            s.append(q)
        time.sleep(BUS_GAP)
    if len(s) < N_SAMPLES * 0.5:
        raise RuntimeError(f"motor_id={motor_id} 读不到回包({len(s)}/{N_SAMPLES});查供电/接线/ID")
    arr = np.asarray(s, dtype=np.float64)
    spread = float(np.max(arr) - np.min(arr))
    q = float(np.mean(arr))
    tag = "  ⚠️抖动大,腿没扶稳?" if spread > 0.05 else ""
    print(f"    q = {q:+.4f}  (抖动 {spread:.4f}){tag}")
    return q


def main() -> None:
    print("=" * 60)
    print("  补 hip_roll 零位 (只改 hip_roll_l / hip_roll_r,其它不动)")
    print("=" * 60)
    print("⚠️  全程零力矩,不驱动电机。")
    print("⚠️  请先把【两腿摆竖直、左右平行、不向内外侧倾】(脚朝正前,膝不内扣不外撇)。")
    input("摆好扶稳后按 Enter 采集(Ctrl+C 取消)...")

    serial = SerialPort(PORT)
    cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    zeros = cfg.get("joints", {}).get("motor_zero_rad")
    if not isinstance(zeros, list) or len(zeros) != 10:
        sys.exit(f"[FATAL] {CONFIG_PATH} 里 motor_zero_rad 不是长度10的列表,先确认标定文件正常")

    for name, mid, idx in HIP_ROLL:
        print(f"\n>>> {name} (ID={mid}, 列表索引 {idx})")
        old = zeros[idx]
        q = sample(serial, mid)
        zeros[idx] = round(q, 4)
        print(f"    motor_zero[{idx}]: {old:+.4f} → {zeros[idx]:+.4f}")

    cfg.setdefault("joints", {})["motor_zero_rad"] = zeros
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"\n✅ 已写入 {CONFIG_PATH}")
    print("   现在 hip_roll 在当前姿态读数应 ≈0。")
    print("   下一步: manual_motor_control.py 看 hip_roll 偏差≈0,再 goto_default 验证不快闪。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断]")
