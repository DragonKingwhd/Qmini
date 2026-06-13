# calib — 关节标定工作流

## 标定模型

```
motor_q = motor_zero + sign × joint_q × 6.33        (real.py / 所有标定脚本共用)
```

| 量 | 什么时候变 | 怎么标 |
|---|---|---|
| `sign` | 接线/机械不变就**永远不变** | 双限位法一次搞定（或 calibrate_sign 逐关节看方向） |
| `motor_zero_rad` | **每次断电作废**（GO 电机 q 相对上电位置） | 每次开机跑 stand_zero.py |

结果都写在 `../config/calibration.yaml` ——这是**每台机器自己的状态文件,不进 git**
（否则每次 pull 都报脏/被 reset 冲掉零位）。首次部署:
`cp config/calibration.example.yaml config/calibration.yaml`。
只要电机不断电,`git pull` 后**不需要**重跑任何标定。

## 三个关键事实（踩过的坑）

1. **URDF joint=0 ≠ 直腿**。`<origin rpy>` 预置了俯仰偏置（hip 1.5 / knee 1.05 /
   ankle 1.22），joint=0 是 CAD 装配的弯腿姿态。
2. **直腿姿态 q_ref = [±0.4, 0, ∓1.5, ±1.05, ∓1.22]**（yaw/roll/hip/knee/ankle，
   左右对称取反）。把直腿当 joint=0 标 → 俯仰零位偏 ~9.5 rad（电机端）→ 发
   DEFAULT 即怼硬限位过流 merror=5。
3. **DEFAULT ≈ 直腿略屈膝**（knee 差 0.05、ankle 差 0.08、roll 差 0.1），所以台架
   姿态验证时初始读数应"接近但不完全等于"DEFAULT。

## 用哪个脚本

| 场景 | 脚本 | 原理 |
|---|---|---|
| **每次开机**（日常） | `stand_zero.py` | 台架直腿姿态 + q_ref 反解 zero。2026-06-13 实测：固件断电**不按**整圈跳，跨断电必走 naive（"⚠️不信任吸附"是正常输出）；2026-06-13 首跑 naive 零位经 goto_default 全✅ |
| 首次标定 / 机械动过 | `capture_zero.py` 选 `a` | 双限位法：顶两头硬限位 + URDF limit 反解 sign+zero，gear_est≈6.33 自检 |
| hip_roll 单独补 | `fix_hip_roll_zero.py` | hip_roll 无真硬限位（双限位齿比 18~20 = 垃圾）；腿竖直 = joint0 直接取 q |
| 验证 sign 方向 | `calibrate_sign.py` | 小幅动每个关节，人眼确认方向 |

所有脚本采集时只发 kp=kd=0（零力矩），不会驱动电机。

## 标完之后

```bash
python3 sim2real/debug/goto_default.py --kp-scale 0.4   # 低增益验证,全关节误差应 <0.1 rad
python3 sim2real/run_qmini.py ...                        # 不断电直接跑
```

⚠️ 标定和运行之间**不能断电**，断电零位作废重标。
⚠️ 头部电机（USB0 ID=0）的 `head.hold_motor_rad` 同样是相对上电位置的，断电后失效。
