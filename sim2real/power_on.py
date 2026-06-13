"""电机断电/重启后的一条龙:重标零位(stand_zero) + 低增益验证(goto DEFAULT)。

GO-M8010-6 断电 = 多圈计数清零 = motor_zero 全部作废(硬属性,无法绕过)。
所以【每次电机上电后】先跑这个,全✅后再跑 run_qmini:

    cd ~/Desktop/Qmini
    # 机器人放台架:两腿伸直竖直、左右平行、脚尖朝前、脚掌放平
    python3 sim2real/power_on.py              # 标定 + 验证,全✅后提示下一步
    python3 sim2real/run_qmini.py --keyboard

只是 git pull / Pi 重启(电机没断电)→ 不需要跑这个。

选项:
    --skip-calib    只验证不重标(零位应该还有效时的快速体检)
    --kp-scale 0.4  验证用增益(默认 0.4,安全软;别在这里用 1.0)
    --hold-secs 3   到 DEFAULT 后保持几秒再卸力
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_SIM2REAL = Path(__file__).resolve().parent
if str(_SIM2REAL) not in sys.path:
    sys.path.insert(0, str(_SIM2REAL))

TOL_RAD = 0.15


def _resolve(p: str) -> Path | None:
    c = Path(p)
    if c.exists():
        return c.resolve()
    alt = _SIM2REAL / p
    return alt.resolve() if alt.exists() else None


def run_stand_zero() -> None:
    """子进程跑 calib/stand_zero.py(交互保持原样),失败即终止。"""
    script = _SIM2REAL / "calib" / "stand_zero.py"
    print("=" * 64)
    print("  [1/2] 重标零位 (calib/stand_zero.py)")
    print("=" * 64)
    rc = subprocess.run([sys.executable, str(script)]).returncode
    if rc != 0:
        sys.exit(f"[FATAL] stand_zero 失败(exit {rc}),不继续验证。")


def verify_goto_default(cfg: Path, kp_scale: float, hold_secs: float) -> bool:
    """goto_default 的验证逻辑:低增益 ramp 到 DEFAULT、保持、出✅/🛑表。"""
    from deploy.constants import DEFAULT_JOINT_POS_VEC, JOINT_NAMES  # noqa: E402
    from deploy.io.real import UnitreeJointDriver  # noqa: E402

    print("\n" + "=" * 64)
    print(f"  [2/2] 低增益验证 (kp_scale={kp_scale}, 去 DEFAULT 保持 {hold_secs}s)")
    print("=" * 64)
    default = np.asarray(DEFAULT_JOINT_POS_VEC, dtype=np.float32)
    joints = UnitreeJointDriver(zero_offset_yaml=str(cfg), kp_scale=kp_scale)

    pos0, _ = joints.read()
    print("\n初始关节角 (calib 换算后):")
    for n, p, d in zip(JOINT_NAMES, pos0, default):
        print(f"  {n:16s} now={p:+.3f}  default={d:+.3f}  Δ={d - p:+.3f}")

    ok = False
    try:
        joints.ramp_to_default(duration_s=6.0)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < hold_secs:
            joints.send_position(default)
            time.sleep(0.02)
        pos, _ = joints.read()
    finally:
        try:
            joints.emergency_stop()
        except Exception as e:
            print(f"[WARN] estop: {e!r}")

    print("\n" + "=" * 64)
    print("  关节            DEFAULT     实测     误差     判定")
    print("=" * 64)
    ok = True
    for n, d, p in zip(JOINT_NAMES, default, pos):
        e = float(p - d)
        good = abs(e) < TOL_RAD
        ok = ok and good
        tag = "✅跟到" if good else "🛑没跟上/零位错"
        print(f"  {n:16s} {d:+.3f}    {p:+.3f}   {e:+.3f}   {tag}")
    print("=" * 64)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/calibration.yaml")
    ap.add_argument("--skip-calib", action="store_true",
                    help="跳过 stand_zero,只做低增益验证")
    ap.add_argument("--kp-scale", type=float, default=0.4)
    ap.add_argument("--hold-secs", type=float, default=3.0)
    args = ap.parse_args()

    if not args.skip_calib:
        run_stand_zero()

    cfg = _resolve(args.config)
    if cfg is None:
        sys.exit(f"[FATAL] calibration not found: {args.config!r}")

    if verify_goto_default(cfg, args.kp_scale, args.hold_secs):
        print("\n✅ 全部跟到,标定有效。下一步(不要断电机电!):")
        print("    python3 sim2real/run_qmini.py --keyboard")
    else:
        sys.exit(
            "\n🛑 有关节没跟到。常见原因:\n"
            "  - 台架姿态没摆准(腿不竖直/脚没放平) → 重摆重跑本脚本\n"
            "  - 又有电机中途重启 → 电机电池断5s重上电,重跑本脚本\n"
            "  - 反复失败 → calib/capture_zero.py 双限位法重标该关节")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断]")
        sys.exit(1)
