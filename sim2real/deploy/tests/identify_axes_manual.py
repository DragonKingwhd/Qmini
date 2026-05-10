"""Manual joint-axis identification.

You move the joint by hand; the script reads motor position to confirm
movement; you answer which axis you saw the leg rotate around.

Workflow:
    1. 机器人吊起来（脚不触地）
    2. 脚本把 4 个未知电机切到 0 力矩（你能用手轻松扳动）
    3. 依次对每个电机：
       - 你按 Enter 后，用手把那个关节朝**单一方向**扳到底再扳回来
       - 脚本看 q 变化方向（确认电机有动）
       - 让你回答：腿绕哪个轴转的（yaw/roll/pitch）
    4. 打印每个电机的 (轴, 方向) 映射

Run:
    cd ~/Desktop/Qmini
    python3 sim2real/deploy/tests/identify_axes_manual.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import List

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

MOTOR_TYPE = MotorType.GO_M8010_6


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
MOVE_THRESHOLD_RAD = 0.3  # motor-side; 0.3 rad ≈ 2.7° joint, easy to clear by hand


def free_motor(serial: SerialPort, motor_id: int) -> MotorData:
    cmd = MotorCmd()
    cmd.motorType = MOTOR_TYPE
    cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = 0.0
    cmd.dq = 0.0
    cmd.tau = 0.0
    cmd.kp = 0.0
    cmd.kd = 0.0
    data = MotorData()
    data.motorType = MOTOR_TYPE
    serial.sendRecv(cmd, data)
    return data


def record_motor(serial: SerialPort, motor_id: int, duration_s: float):
    """Free + log q for `duration_s` seconds. Returns (q_min, q_max, q_start, q_end, n_ok)."""
    t0 = time.perf_counter()
    qs = []
    while time.perf_counter() - t0 < duration_s:
        try:
            data = free_motor(serial, motor_id)
            qs.append(float(data.q))
        except Exception:
            pass
        time.sleep(0.02)  # ~50 Hz
    if not qs:
        return None
    return {
        "q_start": qs[0],
        "q_end": qs[-1],
        "q_min": min(qs),
        "q_max": max(qs),
        "n": len(qs),
    }


def ask_axis() -> str:
    print("    请回答：你看到这条腿绕哪个轴转动？")
    print("      y) yaw   — 水平面内转（八字开合 / 内外旋）")
    print("      r) roll  — 前额面侧倾（向身体外/内倒）")
    print("      p) pitch — 前后摆（前踢/后摆）")
    print("      s) 没看清 / 重测这个")
    while True:
        ans = input("    输入 y/r/p/s: ").strip().lower()
        if ans in ("y", "r", "p", "s"):
            return ans


def ask_sign() -> str:
    """Direction convention question."""
    print("    电机 q 增加（你扳的方向是 q 上升）时，关节往哪一边？")
    print("      + ) 与训练侧约定一致（例如左髋外旋为正）")
    print("      - ) 相反（驱动里要乘 -1）")
    print("      ? ) 现在判断不了，先记 +，部署时再校")
    while True:
        ans = input("    输入 + / - / ?: ").strip()
        if ans in ("+", "-", "?"):
            return ans


def main() -> None:
    print("=" * 64)
    print("  手动关节轴识别")
    print("=" * 64)
    print("⚠️  确认机器人吊起、脚不触地、所有电机均已上电（包括 USB1 上的两颗）。")
    input("按 Enter 继续...")

    serials: dict = {}
    for tgt in TARGETS:
        if tgt.port not in serials:
            serials[tgt.port] = SerialPort(tgt.port)
            print(f"[{tgt.port}] open")

    print("\n→ 把这 4 个电机切到 0 力矩...")
    for tgt in TARGETS:
        try:
            free_motor(serials[tgt.port], tgt.motor_id)
            free_motor(serials[tgt.port], tgt.motor_id)  # send twice
        except Exception as e:
            print(f"  [WARN] {tgt.label}: {e}")
    print("  完成。请验证：用手扳这 4 个关节，应该都很轻松能动。")
    print("  若某个扳不动，去断电重启电机后再来。")

    results = []
    for i, tgt in enumerate(TARGETS, 1):
        print("\n" + "-" * 64)
        print(f"[{i}/{len(TARGETS)}]  {tgt.label}")
        print(f"  操作：用手把这个关节朝**单一方向**扳到边、再扳回来。")
        print(f"  来回扳 1-2 次即可，5 秒。")
        while True:
            input("  按 Enter 开始录制...")
            print(f"  🔴 录制中 {RECORD_S:.0f} 秒... 现在开始扳！")
            stats = record_motor(serials[tgt.port], tgt.motor_id, RECORD_S)
            if stats is None:
                print("  ❌ 完全读不到电机！跳过。")
                results.append((tgt, None, None, None))
                break
            ptp = stats["q_max"] - stats["q_min"]
            net = stats["q_end"] - stats["q_start"]
            print(f"  q 范围: [{stats['q_min']:+.3f}, {stats['q_max']:+.3f}] "
                  f"(峰峰 {ptp:.3f} rad,  净位移 {net:+.3f} rad)")
            if ptp < MOVE_THRESHOLD_RAD:
                print(f"  ⚠️  电机几乎没动 (峰峰 < {MOVE_THRESHOLD_RAD} rad)。")
                print("     可能：扳得太轻 / 这个电机故障没复位 / 0 力矩没生效")
                ans = input("     重测? (y/N): ").strip().lower()
                if ans == "y":
                    continue
            axis = ask_axis()
            if axis == "s":
                print("    重测这个关节。")
                continue
            sign = ask_sign()
            results.append((tgt, axis, sign, stats))
            break

    # summary
    print("\n" + "=" * 64)
    print("  结果总结")
    print("=" * 64)
    axis_full = {"y": "yaw", "r": "roll", "p": "pitch"}
    print(f"{'位置':<24}{'轴':<8}{'方向':<6}{'峰峰(rad,motor)':<18}")
    for tgt, axis, sign, stats in results:
        if axis is None:
            print(f"{tgt.label:<24}{'?':<8}{'?':<6}{'(failed)':<18}")
        else:
            ptp = stats["q_max"] - stats["q_min"]
            print(f"{tgt.label:<24}{axis_full[axis]:<8}{sign:<6}{ptp:<18.3f}")

    # rough joint mapping suggestion
    print("\n建议的关节名映射（结合左/右 + 轴）:")
    name_of = {
        ("左", "yaw"):   "hip_yaw_l",
        ("左", "roll"):  "hip_roll_l",
        ("左", "pitch"): "hip_pitch_l",
        ("右", "yaw"):   "hip_yaw_r",
        ("右", "roll"):  "hip_roll_r",
        ("右", "pitch"): "hip_pitch_r",
    }
    for tgt, axis, sign, _ in results:
        if axis is None:
            continue
        side = "左" if "左" in tgt.label else ("右" if "右" in tgt.label else "?")
        joint = name_of.get((side, axis_full[axis]), "?")
        print(f"  {tgt.port}  ID={tgt.motor_id}  →  {joint}  (sign={sign})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断]")
