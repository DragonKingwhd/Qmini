"""Manual joint-axis identification.

You move the joints by hand; this script just logs.

Workflow:
    1. 机器人吊起来（脚不触地）
    2. 脚本会先把 4 个未知电机切到 0 力矩模式（你应该能轻松扳动）
    3. 脚本依次提示一个电机:
         "请来回扳动 [位置描述]，按 Enter 开始记录 5 秒"
       你按 Enter，然后用手把那个关节**来回小幅摆 2-3 次**（5 秒内），让对应的腿明显地动
    4. 全部 4 个测完后，脚本自动分析每个电机绕哪个轴，并打印总结

Run:
    cd ~/Desktop/Qmini
    python3 sim2real/deploy/tests/identify_axes_manual.py
"""

from __future__ import annotations

import math
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# ---------- unitree SDK ----------
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

# ---------- I2C / IMU ----------
try:
    from smbus2 import SMBus  # type: ignore
except ImportError:
    from smbus import SMBus   # type: ignore

I2C_BUS = 1
MPU_ADDR = 0x68
MPU_PWR_MGMT_1 = 0x6B
MPU_GYRO_CONFIG = 0x1B
MPU_ACCEL_CONFIG = 0x1C
MPU_SMPLRT_DIV = 0x19
MPU_CONFIG = 0x1A
MPU_ACCEL_XOUT_H = 0x3B
GYRO_SCALE = 131.0  # ±250 dps -> deg/s

MOTOR_TYPE = MotorType.GO_M8010_6

# ---------- targets ----------
@dataclass
class Target:
    port: str
    motor_id: int
    label: str

TARGETS: List[Target] = [
    Target("/dev/ttyUSB1", 0, "右髋 (USB1 ID0)"),
    Target("/dev/ttyUSB1", 1, "左髋 (USB1 ID1)"),
    Target("/dev/ttyUSB0", 1, "左胯 (USB0 ID1)"),
    Target("/dev/ttyUSB0", 2, "右胯 (USB0 ID2)"),
]

RECORD_S = 5.0
SAMPLE_HZ = 100.0


# ---------- helpers ----------
def init_mpu(bus: SMBus) -> None:
    bus.write_byte_data(MPU_ADDR, MPU_PWR_MGMT_1, 0x00)
    time.sleep(0.05)
    bus.write_byte_data(MPU_ADDR, MPU_PWR_MGMT_1, 0x01)
    bus.write_byte_data(MPU_ADDR, MPU_SMPLRT_DIV, 0x00)
    bus.write_byte_data(MPU_ADDR, MPU_CONFIG, 0x03)
    bus.write_byte_data(MPU_ADDR, MPU_GYRO_CONFIG, 0x00)
    bus.write_byte_data(MPU_ADDR, MPU_ACCEL_CONFIG, 0x00)
    time.sleep(0.05)


def _i16(hi: int, lo: int) -> int:
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v


def read_gyro(bus: SMBus) -> Tuple[float, float, float]:
    d = bus.read_i2c_block_data(MPU_ADDR, MPU_ACCEL_XOUT_H, 14)
    gx = _i16(d[8], d[9]) / GYRO_SCALE
    gy = _i16(d[10], d[11]) / GYRO_SCALE
    gz = _i16(d[12], d[13]) / GYRO_SCALE
    return gx, gy, gz


def free_motor(serial: SerialPort, motor_id: int, q_hold: float = 0.0) -> MotorData:
    cmd = MotorCmd()
    cmd.motorType = MOTOR_TYPE
    cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = q_hold
    cmd.dq = 0.0
    cmd.tau = 0.0
    cmd.kp = 0.0
    cmd.kd = 0.0
    data = MotorData()
    data.motorType = MOTOR_TYPE
    serial.sendRecv(cmd, data)
    return data


def read_q(serial: SerialPort, motor_id: int) -> float:
    data = free_motor(serial, motor_id)
    return float(data.q)


# ---------- data ----------
@dataclass
class Recording:
    target: Target
    t: List[float] = field(default_factory=list)
    motor_q: List[float] = field(default_factory=list)
    gyro_x: List[float] = field(default_factory=list)
    gyro_y: List[float] = field(default_factory=list)
    gyro_z: List[float] = field(default_factory=list)


def record_one(serial: SerialPort, bus: SMBus, target: Target,
               duration: float, hz: float) -> Recording:
    rec = Recording(target=target)
    dt = 1.0 / hz
    t0 = time.perf_counter()
    while True:
        t = time.perf_counter() - t0
        if t >= duration:
            break
        try:
            q = read_q(serial, target.motor_id)
        except Exception:
            q = float("nan")
        try:
            gx, gy, gz = read_gyro(bus)
        except Exception:
            gx = gy = gz = float("nan")
        rec.t.append(t)
        rec.motor_q.append(q)
        rec.gyro_x.append(gx)
        rec.gyro_y.append(gy)
        rec.gyro_z.append(gz)
        # pace the loop
        elapsed = time.perf_counter() - t0 - t
        sleep_s = dt - elapsed
        if sleep_s > 0:
            time.sleep(sleep_s)
    return rec


# ---------- analysis ----------
def _ptp(arr: List[float]) -> float:
    a = np.asarray(arr, dtype=np.float64)
    a = a[~np.isnan(a)]
    return float(np.ptp(a)) if a.size else 0.0


def _correlation(motor_q: List[float], gyro_axis: List[float]) -> float:
    """correlation of gyro axis with motor_dq (numerical derivative of q)."""
    q = np.asarray(motor_q, dtype=np.float64)
    g = np.asarray(gyro_axis, dtype=np.float64)
    mask = ~(np.isnan(q) | np.isnan(g))
    q = q[mask]; g = g[mask]
    if q.size < 4:
        return 0.0
    dq = np.diff(q)
    g_mid = (g[:-1] + g[1:]) / 2.0
    if dq.std() < 1e-6 or g_mid.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(dq, g_mid)[0, 1])


def analyze(rec: Recording) -> dict:
    motion_rad = _ptp(rec.motor_q)
    ptp = {
        "x": _ptp(rec.gyro_x),
        "y": _ptp(rec.gyro_y),
        "z": _ptp(rec.gyro_z),
    }
    corr = {
        "x": _correlation(rec.motor_q, rec.gyro_x),
        "y": _correlation(rec.motor_q, rec.gyro_y),
        "z": _correlation(rec.motor_q, rec.gyro_z),
    }
    # axis with strongest |corr| wins
    axis = max(corr, key=lambda k: abs(corr[k]))
    axis_name = {"x": "roll", "y": "pitch", "z": "yaw"}[axis]
    sign = "正" if corr[axis] > 0 else "反"
    return {
        "motion_rad": motion_rad,
        "ptp": ptp,
        "corr": corr,
        "axis": axis,
        "axis_name": axis_name,
        "sign": sign,
    }


# ---------- main ----------
def main() -> None:
    print("=" * 64)
    print("  手动关节轴识别")
    print("=" * 64)
    print("⚠️  确认机器人吊起、脚不触地、所有电机均已上电。")
    print("流程：脚本依次提示 4 个未知电机，每个你都用手来回扳 2-3 次。")
    input("\n按 Enter 继续...")

    # open buses
    bus_imu = SMBus(I2C_BUS)
    init_mpu(bus_imu)
    print("[IMU] OK")

    serials = {}
    for tgt in TARGETS:
        if tgt.port not in serials:
            serials[tgt.port] = SerialPort(tgt.port)
            print(f"[{tgt.port}] open")

    # set all targets to free mode
    print("\n→ 把这 4 个电机切到 0 力矩模式...")
    for tgt in TARGETS:
        try:
            free_motor(serials[tgt.port], tgt.motor_id)
        except Exception as e:
            print(f"  [WARN] {tgt.label}: {e}")
    print("  完成。现在你应该可以用手轻松扳动这 4 个关节。")
    print("  若某个关节扳不动，按 Ctrl+C 退出，先排查电机状态。\n")

    recordings: List[Recording] = []
    for i, tgt in enumerate(TARGETS, 1):
        print("-" * 64)
        print(f"[{i}/{len(TARGETS)}]  {tgt.label}")
        print(f"  请准备好用手 **来回扳动** 这个关节（小幅摆动 2-3 次即可）")
        input(f"  按 Enter 开始 {RECORD_S:.0f} 秒录制...")
        print("  🔴 录制中... 现在开始扳动！")
        rec = record_one(serials[tgt.port], bus_imu, tgt, RECORD_S, SAMPLE_HZ)
        print("  ✅ 完成")
        recordings.append(rec)

    # close
    bus_imu.close()

    # analyze
    print("\n" + "=" * 64)
    print("  结果总结")
    print("=" * 64)
    print(f"{'位置':<22} {'motor_Δ(rad)':<14} {'判定轴':<10} {'sign':<6} "
          f"{'corr_x':>7} {'corr_y':>7} {'corr_z':>7}")
    summary = []
    for rec in recordings:
        a = analyze(rec)
        summary.append((rec.target, a))
        print(f"{rec.target.label:<22} {a['motion_rad']:<14.3f} "
              f"{a['axis_name']:<10} {a['sign']:<6} "
              f"{a['corr']['x']:>+7.3f} {a['corr']['y']:>+7.3f} {a['corr']['z']:>+7.3f}")

    # warn if motor didn't move
    print()
    for rec in recordings:
        a = analyze(rec)
        if a["motion_rad"] < 0.05:
            print(f"⚠️  {rec.target.label}: 电机几乎没动 (Δ={a['motion_rad']:.3f}). "
                  "可能在故障态或扳不动；这次结果不可信。")

    print("\n说明：")
    print("  axis_name 的含义（IMU 体坐标系，跟训练侧一致）:")
    print("    roll  → 关节绕 X 轴 (前后) → hip_roll")
    print("    pitch → 关节绕 Y 轴 (左右) → hip_pitch")
    print("    yaw   → 关节绕 Z 轴 (竖直) → hip_yaw")
    print("  sign  = '正' 表示电机q增加方向 与 IMU 同号; '反' 表示反向（驱动里要乘 -1）")
    print("  仅当 |corr| > 0.5 才比较可信；如果三轴 corr 都很小，说明这次扳动太轻或太快。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断]")
