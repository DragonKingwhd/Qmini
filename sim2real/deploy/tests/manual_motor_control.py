"""Interactive keyboard-driven control of all 11 Qmini motors.

Workflow:
    Run script → menu of 11 motors → pick one by number → keyboard control:
        j / k    : -/+ 小步 (0.10 rad motor side, ≈ 0.9° joint side)
        h / l    : -/+ 大步 (0.50 rad motor side, ≈ 4.5° joint side)
        r        : 回到该电机刚进入时的起始位置
        f        : 释放力矩 (free, kp=kd=0)
        p        : 锁定当前位置 (hold, full PD)
        n        : 返回菜单 (释放当前电机)
        x        : 退出整个程序 (释放全部电机)

Safety:
    - 用 pid_sweep.py 验证过的 kp=1.20, kd=0.10
    - target 限位: 离起始位置 ±5.0 rad 电机端 (≈ ±45° 关节端)
    - 退出时所有电机都会被释放 (kp=kd=0)

Run:
    cd ~/Desktop/Qmini
    python3 sim2real/deploy/tests/manual_motor_control.py
"""

from __future__ import annotations

import math
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

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
KP = 1.20
KD = 0.10
LOOP_HZ = 50.0
DT = 1.0 / LOOP_HZ

STEP_SMALL = 0.10   # motor-side rad
STEP_LARGE = 0.30   # was 0.50; reduced to slow down accumulation toward limits
# Max target deviation from q0 (motor-side rad).
# 1.5 rad motor ≈ 0.24 rad ≈ 13.6° joint side. Most URDF joint limits are
# ≥ 0.3 rad on the worst side, so this stays safely inside.
# hip_yaw_l/r have the tightest range (0.8 rad total); even with q0 at the
# midpoint, +/-1.5 rad motor side = +/-13.6° joint = within 30° total.
TARGET_LIMIT_FROM_Q0 = 1.5

class M(NamedTuple):
    port: str       # 串口总线
    mid: int        # 该总线上的电机 ID
    label: str      # constants.JOINT_NAMES 里的名字 (head 除外)
    zh: str         # 中文部位说明 (方便对着实物找)
    pidx: Optional[int]  # 在策略 JOINT_NAMES(10)里的序号; head=None 不参与策略

# 全 11 个电机的映射。pidx 对应 deploy.constants.JOINT_NAMES:
#   0 hip_yaw_l 1 hip_roll_l 2 hip_pitch_l 3 knee_pitch_l 4 ankle_pitch_l
#   5 hip_yaw_r 6 hip_roll_r 7 hip_pitch_r 8 knee_pitch_r 9 ankle_pitch_r
MOTORS: List[M] = [
    M("/dev/ttyUSB0", 0, "head",          "头部 · 不参与策略",            None),
    M("/dev/ttyUSB0", 1, "hip_yaw_l",     "左腿 · 髋偏航(绕竖轴/内外旋)", 0),
    M("/dev/ttyUSB0", 2, "hip_yaw_r",     "右腿 · 髋偏航(绕竖轴/内外旋)", 5),
    M("/dev/ttyUSB1", 0, "hip_roll_r",    "右腿 · 髋横滚(腿内外侧倾)",   6),
    M("/dev/ttyUSB1", 1, "hip_roll_l",    "左腿 · 髋横滚(腿内外侧倾)",   1),
    M("/dev/ttyUSB2", 0, "hip_pitch_r",   "右腿 · 髋俯仰(大腿前后抬)",   7),
    M("/dev/ttyUSB2", 1, "knee_pitch_r",  "右腿 · 膝(小腿前后摆)",       8),
    M("/dev/ttyUSB2", 2, "ankle_pitch_r", "右腿 · 踝(脚掌背屈/跖屈)",    9),
    M("/dev/ttyUSB3", 0, "hip_pitch_l",   "左腿 · 髋俯仰(大腿前后抬)",   2),
    M("/dev/ttyUSB3", 1, "knee_pitch_l",  "左腿 · 膝(小腿前后摆)",       3),
    M("/dev/ttyUSB3", 2, "ankle_pitch_l", "左腿 · 踝(脚掌背屈/跖屈)",    4),
]

# ---- 从 calibration.yaml 读 sign / motor_zero_rad (按 JOINT_NAMES 顺序, 长度10) ----
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "calibration.yaml"


def load_calib() -> Tuple[Optional[list], Optional[list]]:
    """返回 (motor_zero_rad[10], sign[10]); 读不到则 (None, None)。"""
    try:
        import yaml  # noqa: WPS433
        cfg = yaml.safe_load(_CONFIG_PATH.read_text()) or {}
        j = cfg.get("joints", {}) or {}
        return j.get("motor_zero_rad"), j.get("sign")
    except Exception:
        return None, None


def q_is_valid(q: float) -> bool:
    """SDK 没收到回包时 data.q 是未初始化垃圾值(~1e25)。"""
    return math.isfinite(q) and abs(q) < 1.0e4


# ---------- terminal helpers ----------
def setup_raw_terminal():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return old


def restore_terminal(old) -> None:
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)


def get_key_nonblocking() -> str | None:
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


# ---------- motor helpers ----------
def make_cmd(motor_id: int, q: float, kp: float, kd: float) -> MotorCmd:
    cmd = MotorCmd()
    cmd.motorType = MOTOR_TYPE
    cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = float(q)
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


def wake_read(serial: SerialPort, motor_id: int, tries: int = 8) -> Optional[float]:
    """连发若干 0 增益指令唤醒电机(脱离 brake/fault)并读到有效 q。

    扫描菜单时单发一次常超时返回垃圾值;这里重试到拿到合法 q 为止,
    拿不到返回 None(电机未上电/未接/ID 不对)。
    """
    last = None
    for _ in range(tries):
        try:
            q, _ = send(serial, make_cmd(motor_id, 0.0, 0.0, 0.0))
        except Exception:
            q = float("nan")
        if q_is_valid(q):
            last = q
        time.sleep(0.01)
    return last


def wake_status(serial: SerialPort, motor_id: int, tries: int = 8):
    """同 wake_read,但额外返回 merror(电机错误码,非0=故障锁定)和 temp(℃)。
    返回 (q|None, merror|None, temp|None)。"""
    q_out = merr = temp = None
    for _ in range(tries):
        data = MotorData()
        data.motorType = MOTOR_TYPE
        try:
            serial.sendRecv(make_cmd(motor_id, 0.0, 0.0, 0.0), data)
            q = float(data.q)
        except Exception:
            q = float("nan")
        if q_is_valid(q):
            q_out = q
            merr = int(data.merror)
            temp = int(data.temp)
        time.sleep(0.01)
    return q_out, merr, temp


# ---------- per-motor control loop ----------
def control_motor(serial: SerialPort, motor_id: int, label: str) -> str:
    """Returns 'menu' to go back, 'exit' to quit."""
    # initial read with zero gain
    q_now, _ = send(serial, make_cmd(motor_id, 0.0, 0.0, 0.0))
    q0 = q_now
    target = q_now
    mode = "free"  # 'free' or 'hold'

    print(f"\n=== 控制 {label} (ID={motor_id}) ===")
    print(f"  起始 q = {q0:+.3f} rad (motor side)")
    print("  键位:  j/k=-/+小  h/l=-/+大  r=回零  f=放力  p=锁住  n=菜单  x=退出")
    print()

    old_term = setup_raw_terminal()
    last_print = 0.0
    return_code = "menu"
    stale_count = 0
    last_q = q_now
    try:
        while True:
            t = time.perf_counter()

            key = get_key_nonblocking()
            if key:
                if key == "j":
                    target -= STEP_SMALL; mode = "hold"
                elif key == "k":
                    target += STEP_SMALL; mode = "hold"
                elif key == "h":
                    target -= STEP_LARGE; mode = "hold"
                elif key == "l":
                    target += STEP_LARGE; mode = "hold"
                elif key == "r":
                    target = q0; mode = "hold"
                elif key == "f":
                    mode = "free"
                elif key == "p":
                    target = q_now; mode = "hold"
                elif key == "n":
                    return_code = "menu"; break
                elif key == "x":
                    return_code = "exit"; break

                # clamp target
                lo = q0 - TARGET_LIMIT_FROM_Q0
                hi = q0 + TARGET_LIMIT_FROM_Q0
                if target < lo:
                    target = lo
                if target > hi:
                    target = hi

            # send loop tick
            try:
                if mode == "free":
                    q_now, _ = send(serial, make_cmd(motor_id, q_now, 0.0, 0.0))
                else:
                    q_now, _ = send(serial, make_cmd(motor_id, target, KP, KD))
            except Exception:
                pass

            # fault detection: q stuck for too long while target keeps changing
            if mode == "hold" and abs(q_now - last_q) < 1e-4 and abs(q_now - target) > 0.3:
                stale_count += 1
            else:
                stale_count = 0
            last_q = q_now
            if stale_count > 50:   # ~1 second of no movement despite big error
                sys.stdout.write("\n  ⚠️ 电机疑似进入故障态 (q 卡住, err > 0.3 rad). 自动释放并返回菜单.\n")
                sys.stdout.flush()
                return_code = "menu"
                break

            # periodic status line (overwrite same line)
            if t - last_print > 0.1:
                rel_q = q_now - q0
                rel_t = target - q0
                sys.stdout.write(
                    f"\r  q={q_now:+.3f}  target={target:+.3f}  "
                    f"Δq={rel_q:+.3f}  err={q_now-target:+.3f}  mode={mode:4s}    "
                )
                sys.stdout.flush()
                last_print = t

            time.sleep(DT)
    finally:
        # always release motor on exit
        try:
            send(serial, make_cmd(motor_id, q_now, 0.0, 0.0))
        except Exception:
            pass
        restore_terminal(old_term)
        print()
    return return_code


# ---------- menu ----------
def show_menu(serials: Dict[str, SerialPort]) -> None:
    zeros, signs = load_calib()
    print("\n" + "=" * 84)
    print("  Qmini 手动电机控制   (序号 | ID | 关节 | 部位 | 策略# | sign | 零位 | 当前q | 偏差)")
    print("  q=电机端实测  偏差=q-零位  err=merror(电机错误码,非0=故障锁定→需断电重启)")
    print("  '无响应'=未上电/未接/ID不符;  err≠0 → 电机能回数据但不执行指令,必须断电清故障")
    print("=" * 84)
    last_port = None
    n_fault = 0
    for i, m in enumerate(MOTORS):
        if m.port != last_port:
            print(f"\n  ── 总线 {m.port} ──")
            last_port = m.port
        q, merr, temp = wake_status(serials[m.port], m.mid)
        pidx = "  - " if m.pidx is None else f" #{m.pidx} "
        if zeros and signs and m.pidx is not None:
            sgn = f"{signs[m.pidx]:+.0f}"
            z = zeros[m.pidx]
            zs = f"{z:+.3f}"
            ds = f"{q - z:+.3f}" if q is not None else "  -  "
        else:
            sgn, zs, ds = " ? ", "  -  ", "  -  "
        qs = f"{q:+.3f}" if q is not None else "无响应"
        if merr is None:
            es = " - "
        elif merr == 0:
            es = " 0 "
        else:
            es = f"⚠️{merr}"
            n_fault += 1
        ts = f"{temp}℃" if temp is not None else " - "
        print(
            f"  {i:2d} | ID{m.mid} | {m.label:14s} | {m.zh:22s} |"
            f" {pidx} | {sgn} | {zs} | {qs:>8s} | {ds} | err={es} | {ts}"
        )
    if n_fault:
        print(f"\n  🛑 有 {n_fault} 个电机 merror≠0(故障锁定):它们会回数据但不动。")
        print("     必须【电机断电→等5秒→重新上电】清故障;清完 motor_zero 失效需重采。")
    print("\n   x : 退出")


def release_all(serials: Dict[str, SerialPort]) -> None:
    for m in MOTORS:
        try:
            send(serials[m.port], make_cmd(m.mid, 0.0, 0.0, 0.0))
        except Exception:
            pass


def main() -> None:
    ports = sorted({m.port for m in MOTORS})
    serials: Dict[str, SerialPort] = {p: SerialPort(p) for p in ports}
    try:
        while True:
            show_menu(serials)
            choice = input("\n选择电机 (0-10) 或 x 退出: ").strip().lower()
            if choice in ("x", "q", "quit", "exit"):
                break
            try:
                idx = int(choice)
                if not (0 <= idx < len(MOTORS)):
                    raise IndexError
            except (ValueError, IndexError):
                print("  无效输入")
                continue
            m = MOTORS[idx]
            ret = control_motor(serials[m.port], m.mid, f"{m.label} · {m.zh}")
            if ret == "exit":
                break
    finally:
        release_all(serials)
        print("\n所有电机已释放. 退出.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断]")
