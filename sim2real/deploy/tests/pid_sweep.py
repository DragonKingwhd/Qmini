"""Auto-sweep kp/kd for GO-M8010-6 to find safe & well-tracking position-mode gains.

Method:
    For each (kp, kd) combo:
      1. Read current motor q as q0.
      2. Slowly ramp to q0 + delta (motor side, default 0.4 rad ≈ 3.6° joint side)
         with feedforward dq, hold 0.5 s, then ramp back.
      3. Record:
           - final_pos_error    = q_end - q_target  (motor-side, want ~0)
           - max_track_error    = max |actual - commanded| during ramp
           - moved_total        = q max - q min during the whole motion
           - response_ok        = stale-value count (proxy for fault/timeout)

Then prints a table sorted by quality + recommends a (kp, kd) pair.

Setup:
    - Robot吊起、机械上目标关节有 ±0.5 rad 关节端的余量（不能在限位附近）
    - 电机已断电重启过，scan_all_motors.py 干净

Run:
    cd ~/Desktop/Qmini
    python3 sim2real/deploy/tests/pid_sweep.py --port /dev/ttyUSB3 --id 1
        # --id 默认建议: 选膝关节(USB3 ID=1 / USB2 ID=1) 因为限位最宽 (±2.1 rad)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from typing import List

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
LOOP_HZ = 200.0
DT = 1.0 / LOOP_HZ

# small grid; total combos = len(KP_GRID) * len(KD_GRID)
KP_GRID = [0.05, 0.10, 0.20, 0.40, 0.80, 1.50]
KD_GRID = [0.10, 0.30, 0.80, 2.00]


def _make_cmd(motor_id: int, q: float, dq: float, kp: float, kd: float) -> MotorCmd:
    cmd = MotorCmd()
    cmd.motorType = MOTOR_TYPE
    cmd.mode = queryMotorMode(MOTOR_TYPE, MotorMode.FOC)
    cmd.id = motor_id
    cmd.q = float(q)
    cmd.dq = float(dq)
    cmd.tau = 0.0
    cmd.kp = float(kp)
    cmd.kd = float(kd)
    return cmd


def _send(serial: SerialPort, cmd: MotorCmd):
    data = MotorData()
    data.motorType = MOTOR_TYPE
    serial.sendRecv(cmd, data)
    return float(data.q), float(data.dq)


@dataclass
class TrialResult:
    kp: float
    kd: float
    moved: float = 0.0           # max - min over entire trial
    final_err: float = 0.0       # q_end - q_target (motor side)
    max_track_err: float = 0.0   # max |actual - commanded|
    stale_frac: float = 0.0      # fraction of frames where q didn't change between calls
    aborted: bool = False
    note: str = ""


def run_trial(
    serial: SerialPort,
    motor_id: int,
    q0: float,
    delta: float,
    ramp_s: float,
    hold_s: float,
    kp: float,
    kd: float,
) -> TrialResult:
    res = TrialResult(kp=kp, kd=kd)
    q_target = q0 + delta
    v_ff = delta / ramp_s

    # cleanly wake motor first (zero-gain)
    for _ in range(10):
        _send(serial, _make_cmd(motor_id, q0, 0.0, 0.0, 0.0))
        time.sleep(DT)

    qs: List[float] = []
    cmd_qs: List[float] = []

    def step(q_cmd: float, dq_cmd: float) -> float:
        q, _ = _send(serial, _make_cmd(motor_id, q_cmd, dq_cmd, kp, kd))
        return q

    # ramp out
    n = int(ramp_s * LOOP_HZ)
    for i in range(n + 1):
        s = i / n
        q_cmd = q0 + delta * s
        q = step(q_cmd, v_ff)
        qs.append(q); cmd_qs.append(q_cmd)
        time.sleep(DT)
    # hold
    for _ in range(int(hold_s * LOOP_HZ)):
        q = step(q_target, 0.0)
        qs.append(q); cmd_qs.append(q_target)
        time.sleep(DT)
    # ramp back
    for i in range(n + 1):
        s = i / n
        q_cmd = q_target - delta * s
        q = step(q_cmd, -v_ff)
        qs.append(q); cmd_qs.append(q_cmd)
        time.sleep(DT)
    # disable
    _send(serial, _make_cmd(motor_id, q0, 0.0, 0.0, 0.0))

    if not qs:
        res.aborted = True
        res.note = "no samples"
        return res

    res.moved = max(qs) - min(qs)
    res.final_err = qs[len(qs)//2] - q_target  # take mid-hold sample (after ramp, before return)
    res.max_track_err = max(abs(q - c) for q, c in zip(qs, cmd_qs))
    # stale = consecutive identical readings ≥ 5 frames is suspicious
    stale = 0
    run = 1
    last = qs[0]
    for v in qs[1:]:
        if v == last:
            run += 1
        else:
            if run >= 5:
                stale += run
            run = 1
            last = v
    if run >= 5:
        stale += run
    res.stale_frac = stale / len(qs)
    return res


def grade(res: TrialResult, expected_move: float) -> str:
    if res.aborted:
        return "ABORT"
    if res.stale_frac > 0.3:
        return "FAULT"            # motor not responding
    if res.moved < expected_move * 0.3:
        return "STUCK"            # motor barely moved
    if abs(res.final_err) > expected_move * 0.5:
        return "SOFT"             # tracking too poor
    if res.max_track_err > expected_move * 1.5:
        return "OSC?"             # big overshoot
    return "GOOD"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--delta", type=float, default=0.4,
                   help="motor-side step (rad). 0.4 ≈ 3.6° joint with G=6.33")
    p.add_argument("--ramp-s", type=float, default=2.0)
    p.add_argument("--hold-s", type=float, default=0.5)
    args = p.parse_args()

    print("=" * 64)
    print(f"  PID sweep on {args.port} ID={args.id}")
    print(f"  delta={args.delta} rad  ramp={args.ramp_s}s  hold={args.hold_s}s")
    print(f"  Combos: {len(KP_GRID) * len(KD_GRID)}")
    print("=" * 64)
    print("⚠️  确保: 机器人吊起, 该关节有 ±0.5 rad 关节端余量, 电机已重启过.")
    input("按 Enter 开始 ...")

    serial = SerialPort(args.port)
    # initial wake + read q0
    for _ in range(20):
        _send(serial, _make_cmd(args.id, 0.0, 0.0, 0.0, 0.0))
        time.sleep(DT)
    q0, _ = _send(serial, _make_cmd(args.id, 0.0, 0.0, 0.0, 0.0))
    print(f"\nstarting q0 = {q0:+.3f} rad (motor side)")

    results: List[TrialResult] = []
    try:
        for kp in KP_GRID:
            for kd in KD_GRID:
                print(f"\n→ kp={kp:.2f} kd={kd:.2f} ...", end=" ", flush=True)
                res = run_trial(serial, args.id, q0, args.delta,
                                args.ramp_s, args.hold_s, kp, kd)
                tag = grade(res, args.delta)
                print(f"[{tag}] moved={res.moved:.3f} final_err={res.final_err:+.3f} "
                      f"max_dev={res.max_track_err:.3f} stale={res.stale_frac:.2f}")
                results.append(res)

                # detect fault: skip the rest if motor is stuck
                if tag == "FAULT":
                    print("  ⚠️ 电机进入故障态; 中止扫描. 请整机断电重启再试更小的 kp.")
                    break

                # sanity: re-read q0; if drifted, motor not returning
                q_check, _ = _send(serial, _make_cmd(args.id, 0.0, 0.0, 0.0, 0.0))
                if abs(q_check - q0) > 0.3:
                    print(f"  ⚠️ q drift {q_check-q0:+.3f}; pausing 2s ...")
                    time.sleep(2.0)
            else:
                continue
            break
    except KeyboardInterrupt:
        print("\n[中断]")

    # disable on exit
    try:
        _send(serial, _make_cmd(args.id, 0.0, 0.0, 0.0, 0.0))
    except Exception:
        pass

    # summary table
    print("\n" + "=" * 78)
    print(f"{'kp':>6} {'kd':>6}  {'moved':>7} {'final_err':>10} {'max_dev':>8} "
          f"{'stale':>6}  verdict")
    print("-" * 78)
    for r in results:
        tag = grade(r, args.delta)
        print(f"{r.kp:>6.2f} {r.kd:>6.2f}  {r.moved:>7.3f} {r.final_err:>+10.3f} "
              f"{r.max_track_err:>8.3f} {r.stale_frac:>6.2f}  {tag}")

    # recommend: smallest kp giving GOOD; among those, smallest kd
    goods = [r for r in results if grade(r, args.delta) == "GOOD"]
    if goods:
        goods.sort(key=lambda r: (r.kp, r.kd))
        rec = goods[0]
        print("\n推荐: kp = {:.2f}  kd = {:.2f}  "
              "(最小 kp 中跟踪好的)".format(rec.kp, rec.kd))
        print("  其它候选:")
        for r in goods[:5]:
            print(f"    kp={r.kp:.2f} kd={r.kd:.2f}  err={r.final_err:+.3f}")
    else:
        print("\n没有 GOOD 组合. 可能问题:")
        print("  • 所有 kp 都偏低 → 电机跟不动 (SOFT/STUCK) → 加大 kp 上限")
        print("  • 所有 kp 都过流 → 电机进故障 (FAULT) → 减小 delta 重测")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断]")
