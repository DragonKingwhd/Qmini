"""实时打印 body-frame 的 projected_gravity 和 ang_vel,验证 IMU 轴向/符号。

策略靠 projected_gravity(重力方向,指向下)感知自身姿态。轴向/符号一旦错,
策略就分不清前后左右、无法平衡(典型表现:腿在摆但身体往某个方向倒)。

正确约定(body frame: x=前, y=左, z=上;训练侧 _GRAVITY_W=(0,0,-1)):
    竖直静止     → proj_g ≈ ( 0,  0, -1)        z 接近 -1
    机头前倾     → proj_g[0] 变【正】           (前倾=俯)
    向左侧倾     → proj_g[1] 变【正】
    绕竖轴左转   → gyro[2] (wz) 变【正】(逆时针)
    点头(前倾快) → gyro[1] (wy) 变【正】

用法(机器人拿在手里慢慢倾,边看边对):
    cd ~/Desktop/Qmini
    python3 sim2real/debug/check_imu_frame.py

如果某个方向符号反了 → 在 run_qmini 里给 RealIMU 传 axis_sign(对应轴 -1);
如果是两个轴整个对调了 → 传 axis_perm 重排。改完再来这里复验,全对再跑策略。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_SIM2REAL = Path(__file__).resolve().parents[1]
if str(_SIM2REAL) not in sys.path:
    sys.path.insert(0, str(_SIM2REAL))

from deploy.io.real import RealIMU  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--i2c-bus", type=int, default=1)
    # 默认 = real.py RealIMU 的已验证默认(GY-91 绕 y 轴 180° 安装),
    # 所以"不带参数"看到的就是 run_qmini 实际用的。要试别的映射再覆盖。
    ap.add_argument("--perm", type=int, nargs=3, default=[0, 1, 2],
                    help="axis_perm(测试用): 把原始轴重排到 body 轴")
    ap.add_argument("--sign", type=float, nargs=3, default=[-1.0, 1.0, -1.0],
                    help="axis_sign(测试用): 各 body 轴乘 ±1")
    args = ap.parse_args()

    imu = RealIMU(i2c_bus=args.i2c_bus,
                  axis_perm=tuple(args.perm), axis_sign=tuple(args.sign))
    print("\n竖直静止时 proj_g 应 ≈ ( 0, 0, -1)。前倾→g[0]+ 左倾→g[1]+ 左转→wz+")
    print("(--perm / --sign 试出正确映射后,把同样的值填进 run_qmini 的 RealIMU)\n")
    print(f"{'proj_g_x':>9s}{'proj_g_y':>9s}{'proj_g_z':>9s}   "
          f"{'wx':>7s}{'wy':>7s}{'wz':>7s}")
    try:
        while True:
            _lin, ang, g = imu.read()
            print(f"\r{g[0]:+9.3f}{g[1]:+9.3f}{g[2]:+9.3f}   "
                  f"{ang[0]:+7.2f}{ang[1]:+7.2f}{ang[2]:+7.2f}   ",
                  end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[结束]")
    finally:
        imu.close()


if __name__ == "__main__":
    main()
