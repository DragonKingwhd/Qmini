"""Pure 'go to DEFAULT pose and hold' — NO policy / ONNX / IMU / gait.

Isolates the basic motor path: load calibration, gently ramp every joint to
DEFAULT_JOINT_POS at LOW gain, hold, and print per-joint commanded vs measured.

LOW kp (default 0.4×) on purpose: if a joint's sign/zero is wrong and it gets
driven toward a hard stop, low kp just pushes softly (no 40 A stall → no
merror=5 overcurrent), so we can SEE which joints don't reach target safely.

Robot SUSPENDED. Ctrl+C releases all motors.

Run:
    cd ~/Desktop/Qmini
    python3 sim2real/debug/goto_default.py            # kp 0.4, ramp 6s
    python3 sim2real/debug/goto_default.py --kp-scale 0.7 --hold-secs 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_SIM2REAL = Path(__file__).resolve().parents[1]
if str(_SIM2REAL) not in sys.path:
    sys.path.insert(0, str(_SIM2REAL))

from deploy.constants import DEFAULT_JOINT_POS_VEC, JOINT_NAMES  # noqa: E402
from deploy.io.real import UnitreeJointDriver  # noqa: E402


def _resolve(p: str) -> Path | None:
    c = Path(p)
    if c.exists():
        return c.resolve()
    alt = _SIM2REAL / p
    return alt.resolve() if alt.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/calibration.yaml")
    ap.add_argument("--ramp-secs", type=float, default=6.0)
    ap.add_argument("--hold-secs", type=float, default=0.0,
                    help="0 = 一直保持直到 Ctrl+C(默认);>0 = 保持N秒后卸力")
    ap.add_argument("--kp-scale", type=float, default=0.4,
                    help="Gain scale (low = safe; raise once it tracks).")
    ap.add_argument("--bus-gap", type=float, default=0.0006)
    ap.add_argument("--ankle-trim-l", type=float, default=0.0,
                    help="左踝目标加偏(关节rad,调到左脚平贴。脚尖朝下→正负各试).")
    ap.add_argument("--ankle-trim-r", type=float, default=0.0,
                    help="右踝目标加偏(关节rad).")
    args = ap.parse_args()

    cfg = _resolve(args.config)
    if cfg is None:
        sys.exit(f"[FATAL] calibration not found: {args.config!r} "
                 f"(tried cwd and {_SIM2REAL})")
    print(f"[INFO] calibration: {cfg}")
    print(f"[INFO] kp_scale={args.kp_scale}  ramp={args.ramp_secs}s  "
          f"hold={args.hold_secs}s  (NO policy/IMU/ONNX)")

    default = np.asarray(DEFAULT_JOINT_POS_VEC, dtype=np.float32)
    # ankle_pitch_l=idx4, ankle_pitch_r=idx9。trim 只改保持目标,用来把脚调平;
    # 找到值后再固化进 calibration motor_zero(见末尾提示)。
    target = default.copy()
    if args.ankle_trim_l or args.ankle_trim_r:
        target[4] += args.ankle_trim_l
        target[9] += args.ankle_trim_r
        print(f"[trim] 左踝{args.ankle_trim_l:+.3f} 右踝{args.ankle_trim_r:+.3f} (关节rad)")
    joints = UnitreeJointDriver(zero_offset_yaml=str(cfg),
                                bus_gap_s=args.bus_gap,
                                kp_scale=args.kp_scale)

    pos0, _ = joints.read()
    print("\n初始关节角 (calib 换算后):")
    for n, p, d in zip(JOINT_NAMES, pos0, default):
        print(f"  {n:16s} now={p:+.3f}  default={d:+.3f}  Δ={d - p:+.3f}")

    try:
        print(f"\n[ramp] 缓慢 {args.ramp_secs}s 去 DEFAULT (低增益,安全)...")
        joints.ramp_to_default(duration_s=args.ramp_secs)
        if args.hold_secs > 0:
            print(f"[hold] 保持 DEFAULT {args.hold_secs}s 后自动卸力...")
        else:
            print("[hold] 持续保持 DEFAULT —— 按 Ctrl+C 结束卸力")
        t0 = time.perf_counter()
        last = 0.0
        while args.hold_secs <= 0 or (time.perf_counter() - t0 < args.hold_secs):
            joints.send_position(target)
            t = time.perf_counter()
            if t - last > 0.5:
                pos, _ = joints.read()
                errs = np.abs(pos - target)
                worst = int(np.argmax(errs))
                print(f"  t={t - t0:5.1f}s  max|err|={errs.max():.3f} "
                      f"@{JOINT_NAMES[worst]}")
                last = t
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        try:
            joints.emergency_stop()
        except Exception as e:
            print(f"[WARN] estop: {e!r}")

    pos, _ = joints.read()
    print("\n" + "=" * 70)
    print("  关节         DEFAULT     实测     误差     判定")
    print("=" * 70)
    for n, d, p in zip(JOINT_NAMES, default, pos):
        e = p - d
        tag = "✅跟到" if abs(e) < 0.15 else (
            "⚠️差一点" if abs(e) < 0.5 else "🛑没跟上/方向反/卡死")
        print(f"  {n:16s} {d:+.3f}    {p:+.3f}   {e:+.3f}   {tag}")
    print("=" * 70)
    print("全✅ → 电机路径没问题,问题在策略侧;有🛑 → 那几个关节标定(sign/zero)还不对")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断]")
