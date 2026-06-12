"""台架直腿姿态一键零位 — 每次上电后重标 motor_zero 的日常脚本。

为什么需要它:GO-M8010-6 的输出 q 是【相对上电位置】的(转子单圈绝对编码器,
圈数断电清零),所以 motor_zero 每次断电都作废,每次上电都要重标。
双限位法(capture_zero 的 a)最准但要逐关节顶限位,太慢;本脚本用台架上
"两腿伸直竖直、左右平行、脚尖朝前、脚掌放平"这一个可重复姿态,一次采全 10 个。

参考姿态的关节角(q_ref)——不是 0,也不完全是 DEFAULT!
URDF <origin rpy> 给每个关节预置了偏置,直腿姿态 = 每个关节"净旋转≈0":

    关节        origin(rpy)   axis      直腿 q_ref     DEFAULT   差
    hip_yaw_l   yaw +0.4      0 0 -1    +0.40          +0.40     0
    hip_roll_l  0             1 0 0      0.00          -0.10     0.10
    hip_pitch_l pitch +1.5    0 +1 0    -1.50          -1.50     0
    knee_l      pitch +1.05   0 -1 0    +1.05          +1.00     0.05
    ankle_l     pitch +1.22   0 +1 0    -1.22          -1.30     0.08
    (右腿对称取反)

老 z 法把这个姿态当 joint=0 → 俯仰零位偏 ~9.5 rad(电机端)→ 过流。本脚本
用上面的 q_ref 反解:motor_zero = q_raw − sign × q_ref × 6.33(sign 从
calibration.yaml 读,接线不变 sign 不变,断电不丢)。

精度增强(吸附):上次标定的 zero 还在 yaml 里时,若断电后 q 恰好跳 k×2π
(转子单圈绝对编码器的行为),把 naive zero 吸附到 old_zero + k×2π 就能继承
上次双限位标定的精度——摆姿态只要准到 ±0.25 rad(关节端 ±14°)就够。

但固件是否真按整圈跳【未实测确认】(此前只观察到"每次上电零位都变")。
所以吸附加了集体表决门:整圈假设成立时 10 个关节的残差=摆姿态误差,必然
集体小;若是任意复位,残差在 ±π 均匀散布,必然集体大。≥8 个关节残差 ≤ 容限
才启用吸附,否则全部回退 naive(纯姿态反解)并明说。首次跑完看打印就知道
你的固件属于哪种——属于整圈跳的话以后吸附常开,白赚双限位精度。

安全:全程只发 kp=kd=0(零力矩),绝不驱动电机。

用法(机器人放台架上摆好直腿,先上电再运行):
    cd ~/Desktop/Qmini
    python3 sim2real/calib/stand_zero.py              # 标完直接写 yaml(自动备份)
    python3 sim2real/calib/stand_zero.py --dry-run    # 只看不写
    python3 sim2real/calib/stand_zero.py --naive      # 不吸附,纯姿态反解

标完验证(不断电!):
    python3 sim2real/debug/goto_default.py --kp-scale 0.4
"""

from __future__ import annotations

import argparse
import math
import shutil
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

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "sim2real/config/calibration.yaml"
MOTOR_TYPE = MotorType.GO_M8010_6
GEAR_RATIO = 6.33
TWO_PI = 2.0 * math.pi

JOINT_NAMES = [
    "hip_yaw_l", "hip_roll_l", "hip_pitch_l", "knee_pitch_l", "ankle_pitch_l",
    "hip_yaw_r", "hip_roll_r", "hip_pitch_r", "knee_pitch_r", "ankle_pitch_r",
]

# (port, motor_id) per JOINT_NAMES entry — same mapping as real.py / capture_zero.py.
JOINT_PORTS: List[Tuple[str, int]] = [
    ("/dev/ttyUSB0", 1), ("/dev/ttyUSB1", 1), ("/dev/ttyUSB3", 0),
    ("/dev/ttyUSB3", 1), ("/dev/ttyUSB3", 2), ("/dev/ttyUSB0", 2),
    ("/dev/ttyUSB1", 0), ("/dev/ttyUSB2", 0), ("/dev/ttyUSB2", 1),
    ("/dev/ttyUSB2", 2),
]

# 直腿参考姿态(见文件头推导表)。q1.urdf 的 origin rpy / axis 改了才需要更新。
Q_REF = [0.40, 0.00, -1.50, 1.05, -1.22, -0.40, 0.00, 1.50, -1.05, 1.22]

# 吸附残差容限(电机端 rad)。姿态误差 0.16 rad(关节端) × 6.33 ≈ 1.0;
# 真正的整圈跳变是 2π≈6.28 的整数倍,1.0 的容限分得很开。
SNAP_TOL_MOTOR_RAD = 1.0

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


def sample_all(serials) -> Optional[List[float]]:
    """零力矩采全 10 个关节的 motor q(平均 N_SAMPLES 次)。"""
    print("    唤醒(零力矩)...")
    for _ in range(WAKE_ITERS):
        for port, mid in JOINT_PORTS:
            try:
                read_q(serials[port], mid)
            except Exception:
                pass
        time.sleep(0.01)
    print(f"    采集 {N_SAMPLES} 次取平均,保持机器人不动...")
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
            print(f"    ❌ {name}: 读不到回包({len(acc[i])}/{N_SAMPLES}),查供电/接线")
            return None
        arr = np.asarray(acc[i], dtype=np.float64)
        spread = float(np.max(arr) - np.min(arr))
        if spread > 0.05:
            print(f"    ⚠️ {name}: 读数抖动 {spread:.3f} rad,机器人没扶稳?")
        out.append(float(np.mean(arr)))
    return out


def _list10(cfg: dict, key: str) -> Optional[List[float]]:
    v = (cfg.get("joints", {}) or {}).get(key)
    if isinstance(v, list) and len(v) == len(JOINT_NAMES):
        return [float(x) for x in v]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="只打印结果,不写 yaml")
    ap.add_argument("--naive", action="store_true",
                    help="不吸附旧零位,纯按姿态反解(姿态精度=零位精度)")
    ap.add_argument("--snap-tol", type=float, default=SNAP_TOL_MOTOR_RAD,
                    help="吸附残差容限(电机端 rad,默认 1.0)")
    args = ap.parse_args()

    print("=" * 64)
    print("  台架直腿一键零位 (stand_zero)")
    print("=" * 64)
    print("姿态要求:两腿【完全伸直、竖直向下、左右平行】、脚尖朝正前、脚掌放平")
    print("(= 直腿参考姿态 q_ref,≠ URDF joint=0 的弯腿姿态)")
    print(f"读写: {CONFIG_PATH}")

    cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    signs = _list10(cfg, "sign")
    if signs is None:
        sys.exit("[FATAL] calibration.yaml 里没有 joints.sign(长度10)。\n"
                 "  sign 只需标一次(接线不变就不变):先跑双限位法 capture_zero.py")
    old_zeros = _list10(cfg, "motor_zero_rad")
    if old_zeros is None and not args.naive:
        print("[INFO] yaml 里没有旧 motor_zero_rad → 本次全部用姿态反解(naive)")

    input("\n摆好扶稳后按 Enter 开始采集(Ctrl+C 取消)...")

    ports = sorted({p for p, _ in JOINT_PORTS})
    serials = {p: SerialPort(p) for p in ports}
    raw = sample_all(serials)
    if raw is None:
        sys.exit("[FATAL] 采集失败,未写任何文件")

    # 先全部算 naive + 吸附候选,再集体表决是否信任"断电跳 k×2π"假设
    naives: List[float] = []
    snappeds: List[Optional[float]] = []
    resids: List[Optional[float]] = []
    for i in range(len(JOINT_NAMES)):
        naive = raw[i] - signs[i] * Q_REF[i] * GEAR_RATIO
        naives.append(naive)
        if old_zeros is not None and not args.naive:
            k = round((naive - old_zeros[i]) / TWO_PI)
            snappeds.append(old_zeros[i] + TWO_PI * k)
            resids.append(naive - snappeds[i])
        else:
            snappeds.append(None)
            resids.append(None)

    n_small = sum(1 for r in resids if r is not None and abs(r) <= args.snap_tol)
    snap_trusted = (old_zeros is not None and not args.naive and n_small >= 8)

    print(f"\n  {'joint':14s} {'raw_q':>8s} {'naive':>9s} "
          f"{'snapped':>9s} {'残差':>7s} {'采用':>9s}  状态")
    new_zeros: List[float] = []
    n_snap = 0
    for i, name in enumerate(JOINT_NAMES):
        use, tag = naives[i], "naive"
        snap_str = f"{snappeds[i]:+.4f}" if snappeds[i] is not None else "-"
        resid_str = f"{resids[i]:+.3f}" if resids[i] is not None else "-"
        if snap_trusted and resids[i] is not None and abs(resids[i]) <= args.snap_tol:
            use, tag = snappeds[i], "✅吸附"
            n_snap += 1
        elif snap_trusted:
            tag = "⚠️残差大→naive(该关节机械动过/没摆准?)"
        new_zeros.append(round(float(use), 4))
        print(f"  {name:14s} {raw[i]:+8.3f} {naives[i]:+9.4f} "
              f"{snap_str:>9s} {resid_str:>7s} {new_zeros[i]:+9.4f}  {tag}")

    if old_zeros is not None and not args.naive:
        if snap_trusted:
            print(f"\n  整圈跳变假设成立({n_small}/10 残差≤{args.snap_tol})"
                  f" → 吸附 {n_snap} 个,继承双限位精度")
        else:
            print(f"\n  ⚠️ 只有 {n_small}/10 关节残差≤{args.snap_tol} → 不信任吸附,"
                  f"全部用 naive(精度=摆姿态精度)")
            print("     可能:固件断电不按整圈跳 / 姿态整体摆偏 / 旧零位本来就错。")
            print("     先 goto_default --kp-scale 0.4 低增益验证;有怀疑重跑双限位法。")

    if args.dry_run:
        print("\n[dry-run] 未写文件")
        return

    bak = CONFIG_PATH.with_suffix(".yaml.bak")
    shutil.copy2(CONFIG_PATH, bak)
    cfg.setdefault("joints", {})
    cfg["joints"]["motor_zero_rad"] = new_zeros
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"\n✅ motor_zero_rad 已写入 {CONFIG_PATH}(sign 未动,旧文件备份在 {bak.name})")
    print("下一步(不要断电!断电零位又作废):")
    print("    python3 sim2real/debug/goto_default.py --kp-scale 0.4")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断] 未写任何文件")
