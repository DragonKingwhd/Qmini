"""Two-hard-stop calibration — solves motor_zero AND sign per joint, no guessing.

Deploy model (real.py): motor_q = zero + sign * joint_q * GEAR_RATIO.
Push a joint to BOTH its mechanical hard stops; the joint angle there is the
known URDF limit. Two readings + two known angles → solve exactly:

    q_lo = zero + sign * LOWER * G        (joint held at lower hard stop)
    q_hi = zero + sign * UPPER * G        (joint held at upper hard stop)
    D = q_hi - q_lo = sign * (UPPER-LOWER) * G
      sign      = +1 if D > 0 else -1          (UPPER>LOWER, G>0)
      gear_est  = |D| / (UPPER-LOWER)          (sanity: should ≈ 6.33)
      zero      = q_lo - sign * LOWER * G

No guessing which stop / which sign; gear_est self-validates each joint
(far from 6.33 ⇒ didn't reach a true hard stop / wiring issue ⇒ redo).
Independent of power-cycle-relative q AS LONG AS you don't power-cycle
between this calibration and the run.

SAFETY: ONLY ever sends kp=kd=0 (zero torque). Never drives the motors.
Robot SUSPENDED. Hold the joint FIRMLY against each hard stop while sampling.

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
GEAR_RATIO = 6.33

JOINT_NAMES = [
    "hip_yaw_l", "hip_roll_l", "hip_pitch_l", "knee_pitch_l", "ankle_pitch_l",
    "hip_yaw_r", "hip_roll_r", "hip_pitch_r", "knee_pitch_r", "ankle_pitch_r",
]

JOINT_ZH = [
    "左腿·髋偏航(绕竖轴/内外旋)", "左腿·髋横滚(腿内外侧倾)",
    "左腿·髋俯仰(大腿前后抬)", "左腿·膝(小腿前后摆)",
    "左腿·踝(脚掌背屈/跖屈)", "右腿·髋偏航(绕竖轴/内外旋)",
    "右腿·髋横滚(腿内外侧倾)", "右腿·髋俯仰(大腿前后抬)",
    "右腿·膝(小腿前后摆)", "右腿·踝(脚掌背屈/跖屈)",
]

# (port, motor_id) per JOINT_NAMES entry — same mapping as real.py.
JOINT_PORTS: List[Tuple[str, int]] = [
    ("/dev/ttyUSB0", 1), ("/dev/ttyUSB1", 1), ("/dev/ttyUSB3", 0),
    ("/dev/ttyUSB3", 1), ("/dev/ttyUSB3", 2), ("/dev/ttyUSB0", 2),
    ("/dev/ttyUSB1", 0), ("/dev/ttyUSB2", 0), ("/dev/ttyUSB2", 1),
    ("/dev/ttyUSB2", 2),
]

# URDF q1.urdf <limit lower/upper> per JOINT_NAMES — the known joint angles
# at each mechanical hard stop.
LOWER = [-0.10, -0.30, -2.10,  0.00, -2.50, -0.70, -0.60,  0.00, -2.10,  0.00]
UPPER = [ 0.70,  0.60,  0.00,  2.10,  0.00,  0.10,  0.30,  2.10,  0.00,  2.50]

N_SAMPLES = 30
WAKE_ITERS = 10


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


def _list10(cfg: dict, key: str) -> List[Optional[float]]:
    v = (cfg.get("joints", {}) or {}).get(key)
    if isinstance(v, list) and len(v) == len(JOINT_NAMES):
        return [float(x) for x in v]
    return [None] * len(JOINT_NAMES)


def write_calib(zeros: List[Optional[float]], signs: List[Optional[float]]) -> None:
    cfg = load_cfg()
    old_z = _list10(cfg, "motor_zero_rad")
    old_s = _list10(cfg, "sign")
    mz = [zeros[i] if zeros[i] is not None else old_z[i]
          for i in range(len(JOINT_NAMES))]
    sg = [signs[i] if signs[i] is not None else old_s[i]
          for i in range(len(JOINT_NAMES))]
    mz = [0.0 if v is None else float(v) for v in mz]
    sg = [1.0 if v is None else float(v) for v in sg]
    cfg.setdefault("joints", {})
    cfg["joints"]["motor_zero_rad"] = mz
    cfg["joints"]["sign"] = sg
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def sample_stop(serials, idx: int, which: str) -> Optional[float]:
    """Sample averaged q while the joint is held hard against one stop."""
    port, mid = JOINT_PORTS[idx]
    for _ in range(WAKE_ITERS):
        try:
            read_q(serials[port], mid)
        except Exception:
            pass
        time.sleep(0.01)
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
        print(f"    ❌ {which}: 读不到回包({len(s)}/{N_SAMPLES});查供电/接线/ID")
        return None
    arr = np.asarray(s, dtype=np.float64)
    spread = float(np.max(arr) - np.min(arr))
    q = float(np.mean(arr))
    tag = "  ⚠️抖动大,没顶死?" if spread > 0.05 else ""
    print(f"    {which}: q = {q:+.4f}  (抖动 {spread:.4f}){tag}")
    return q


def capture_joint(serials, idx: int):
    """Two-stop solve. Returns (zero, sign, gear_est) or None."""
    name, zh = JOINT_NAMES[idx], JOINT_ZH[idx]
    lo, hi = LOWER[idx], UPPER[idx]
    print(f"\n>>> [{name}] {zh}")
    print(f"    顶到【下限】那头 (关节角 ≈ {lo:+.2f} rad),用手/工具死死抵住")
    input("    抵住后按 Enter 采下限(Ctrl+C 取消本关节)...")
    q_lo = sample_stop(serials, idx, "下限")
    if q_lo is None:
        return None
    print(f"    现在顶到【上限】那头 (关节角 ≈ {hi:+.2f} rad),死死抵住")
    input("    抵住后按 Enter 采上限...")
    q_hi = sample_stop(serials, idx, "上限")
    if q_hi is None:
        return None

    span = hi - lo                       # > 0 (URDF upper>lower)
    D = q_hi - q_lo                      # = sign*span*G
    gear_est = abs(D) / span
    sign = 1.0 if D > 0 else -1.0
    zero = round(q_lo - sign * lo * GEAR_RATIO, 4)
    ok = abs(gear_est - GEAR_RATIO) < 1.0
    flag = "✅" if ok else "⚠️齿比偏离6.33,可能没顶到真硬限位/接线问题→建议重做"
    print(f"    → sign={sign:+.0f}  zero={zero:+.4f}  齿比实测={gear_est:.2f} {flag}")
    return zero, sign, gear_est


def capture_zero_pose_all(serials) -> Optional[List[float]]:
    """几何零位法:所有关节同时摆到 URDF joint_q=0(腿伸直竖直平行、脚平),
    一次性采全 10 个 → motor_zero = q(与 sign/齿比/限位都无关,最干净;
    hip_roll 也一并解决)。返回 10 个 zero,失败返回 None。"""
    print("\n>>> 几何零位姿态:两腿【完全伸直、竖直向下、左右平行、脚掌放平】")
    print("    = URDF joint_q=0。摆好扶稳(越准越好,小偏差可后期 offset 微调)")
    input("    摆好后按 Enter 一次性采全部 10 个(Ctrl+C 取消)...")
    print("    唤醒(零力矩)...")
    for _ in range(WAKE_ITERS):
        for port, mid in JOINT_PORTS:
            try:
                read_q(serials[port], mid)
            except Exception:
                pass
        time.sleep(0.01)
    print(f"    采集 {N_SAMPLES} 次取平均,保持不动...")
    acc: List[List[float]] = [[] for _ in JOINT_NAMES]
    for _ in range(N_SAMPLES):
        for i, (port, mid) in enumerate(JOINT_PORTS):
            try:
                q = read_q(serials[port], mid)
            except Exception:
                q = float("nan")
            if q_ok(q):
                acc[i].append(q)
        time.sleep(0.01)
    out: List[float] = []
    for i, name in enumerate(JOINT_NAMES):
        if len(acc[i]) < N_SAMPLES * 0.5:
            print(f"    ❌ {name}: 读不到回包,放弃本次(查供电/接线)")
            return None
        arr = np.asarray(acc[i], dtype=np.float64)
        z = round(float(np.mean(arr)), 4)
        spread = float(np.max(arr) - np.min(arr))
        tag = " ⚠️抖" if spread > 0.05 else ""
        print(f"    {name:14s} motor_zero = {z:+.4f} (抖{spread:.3f}){tag}")
        out.append(z)
    return out


def main() -> None:
    print("=" * 60)
    print("  标定: z=几何零位法(推荐) / 序号=双限位法(解sign用)")
    print("=" * 60)
    print("⚠️  机器人【悬空】。全程零力矩,不驱动电机。")
    print("⚠️  每关节顶两头硬限位各采一次;顶时用力抵死别松。")
    print(f"⚠️  解出即写 {CONFIG_PATH} (motor_zero_rad + sign)")

    ports = sorted({p for p, _ in JOINT_PORTS})
    serials = {p: SerialPort(p) for p in ports}

    cfg = load_cfg()
    base_z = _list10(cfg, "motor_zero_rad")
    base_s = _list10(cfg, "sign")
    zeros: List[Optional[float]] = [None] * len(JOINT_NAMES)
    signs: List[Optional[float]] = [None] * len(JOINT_NAMES)
    gears: List[Optional[float]] = [None] * len(JOINT_NAMES)

    while True:
        print("\n" + "-" * 72)
        for i, n in enumerate(JOINT_NAMES):
            port, mid = JOINT_PORTS[i]
            if zeros[i] is not None:
                _sg = f"{signs[i]:+.0f}" if signs[i] is not None else (
                    f"{base_s[i]:+.0f}(旧)" if base_s[i] is not None else "?")
                _gr = f" 齿比={gears[i]:.2f}" if gears[i] is not None else ""
                st = f"✅ zero={zeros[i]:+.3f} sign={_sg}{_gr}"
            elif base_z[i] is not None:
                st = f"旧 zero={base_z[i]:+.3f}(未重标)"
            else:
                st = "未标"
            print(f"  {i:2d} {JOINT_ZH[i]:22s} {n:14s} {port} ID={mid}  {st}")
        print("  z=几何零位法一次采全部(motor_zero=q, sign保持)  ← 推荐")
        print("  序号/a=双限位法(主要用来解 sign)  w=写入退出  q=退出")
        choice = input("z / 序号 / a / w / q: ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("退出。已写入的保留。")
            return
        if choice == "z":
            try:
                zs = capture_zero_pose_all(serials)
            except KeyboardInterrupt:
                print("\n  取消")
                continue
            if zs is not None:
                for i in range(len(JOINT_NAMES)):
                    zeros[i] = zs[i]
                    gears[i] = None
                # sign 保持现有(传 None → write_calib 保留旧 sign)
                write_calib(zeros, [None] * len(JOINT_NAMES))
                print(f"✅ 几何零位已写入 {CONFIG_PATH}(sign 保持不变)")
            continue
        if choice == "w":
            write_calib(zeros, signs)
            print(f"✅ 已写入 {CONFIG_PATH}")
            return
        if choice == "a":
            for i in range(len(JOINT_NAMES)):
                try:
                    r = capture_joint(serials, i)
                except KeyboardInterrupt:
                    print("\n  跳过该关节")
                    continue
                if r is not None:
                    zeros[i], signs[i], gears[i] = r
                    write_calib(zeros, signs)
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
            r = capture_joint(serials, idx)
        except KeyboardInterrupt:
            print("\n  本次取消")
            continue
        if r is not None:
            zeros[idx], signs[idx], gears[idx] = r
            write_calib(zeros, signs)
            print(f"    ✅ 已写入 {JOINT_NAMES[idx]}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断]")
