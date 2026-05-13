# Qmini Sim2Real 部署进度记录

> 我在桌面端能做的代码骨架先全部写完，所有跟实物相关的参数/方向/单位都先用合理默认值。
> 你晚上回去拿到机器人后照着「待你实物验证」清单逐项核对。

---

## ✅ 已完成（2026-05-13）

### 1. `deploy/io/real.py` 补全
原文件只有 `UnitreeJointDriver`（电机已是完成态）。这次新增：

- **`RealIMU`** —— GY-91 (MPU9250) 的 I2C 驱动
  - 读取 14 字节 burst（accel + gyro）
  - 单位：accel → g 单位 → 归一化成 `proj_gravity` 单位向量；gyro deg/s → rad/s
  - 支持 `set_gyro_bias()`（被 `controller.calibrate_imu()` 调用，开机 3 秒静态采集）
  - **`base_lin_vel` 暂时返回 `zeros(3)`**（训练侧这通道有 Unoise，先靠 sim2real margin 顶住；后续如果飘可以加状态估计器）
  - 提供 `axis_perm` / `axis_sign` 参数做轴重映射，默认 identity

- **`JoystickCommand`** —— pygame 读手柄 → `[vx, vy, wz]`
  - 默认 Xbox-style 摇杆映射：左摇杆 Y→vx、左摇杆 X→vy、右摇杆 X→wz
  - 默认 scale: vx=0.4, vy=0.3, wz=1.0
  - 死区 0.10
  - 没插手柄时返回零（机器人停在原地）

### 2. `run_qmini.py` 切到真驱动
- 默认用 `UnitreeJointDriver + RealIMU + JoystickCommand`
- `--mock` 标志：桌面端干跑测试
- `--constant-cmd`：不接手柄时用 `--vx/--vy/--wz` 跑恒定速度（README 推荐的渐进起步方式）

### 3. 文件清单
```
deploy/io/real.py         ← 新增 RealIMU + JoystickCommand（电机部分原有）
run_qmini.py              ← 改写，默认 real，--mock 回退
config/calibration.yaml   ← 已有，gyro_bias 还是 null（开机会自动标定）
record.md                 ← 本文件
```

---

## ⚠️ 待你实物验证 / 调整的项

### A. IMU 轴方向（**最关键**）
当前默认 `axis_perm=(0,1,2)`, `axis_sign=(1,1,1)`。

**验证方法**：在 Pi 上单独跑 `python3 -m deploy.tests.test_imu_gy91` 然后：
- 机器人**保持直立** → `proj_g` 应该接近 `[0, 0, +1]`
- 机器人**前倾**（pitch forward） → `proj_g[0]` 应该变正
- 机器人**左倾**（roll left） → `proj_g[1]` 应该变正
- 机器人**绕竖轴顺时针转**（俯视） → `gyro[2]` 应该变**负**（右手系）

如果某轴方向不对，改 `run_qmini.py` 的 `_build_real()` 里：
```python
imu = RealIMU(axis_perm=(1, 0, 2), axis_sign=(1, -1, 1))  # 举例
```

### B. 关节 sign 标定
`config/calibration.yaml` 里 `joints.sign` 全是 +1（占位）。

**验证方法**：
1. 先**只发 `DEFAULT_JOINT_POS`** 看机器人能否平静站立
2. 用 `deploy/tests/calibrate_sign.py`（已存在）一对一发 +0.1 rad，看每个关节是不是朝 URDF 正方向走
3. 不对就把对应位置改成 -1.0

### C. 手柄按键映射
当前假设 Xbox 风格：axis 0=左X、axis 1=左Y、axis 3=右X。
**很多手柄不一样**（北通、PS、Switch Pro 各家不同）。

**验证方法**：先插着手柄跑 `python3 -c "import pygame; pygame.init(); ..."` 打印 axes，或者直接看 README 里 pygame 摇杆调试代码。我建议你先用 `--constant-cmd --vx 0.0` 走通流程，**第一次跑别开手柄**。

### D. ONNX 模型路径
`run_qmini.py --onnx policy.onnx` 默认从当前目录读。你需要：
- 把训练导出的 `.onnx` 拷到 Pi 的 `~/qmini/sim2real/policy.onnx`
- 或者启动时显式传 `--onnx /path/to/policy.onnx`

### E. `unitree_actuator_sdk` 路径
`real.py:38` 写死了 `/home/pi/unitree_actuator_sdk/lib`。Pi 上如果 SDK 装在别处需要改。

### F. I2C 启用
确认 Pi 上 I2C 已启用：
```bash
sudo raspi-config   # Interface Options → I2C → Enable
sudo apt install -y i2c-tools python3-smbus
pip3 install smbus2 pygame
i2cdetect -y 1      # 应该看到 0x68 (MPU9250) 和 0x76/0x77 (BMP280)
```

---

## 🧪 建议的上电验证顺序

> **每一步都先确认通过再做下一步。** 哪一步异常就停下来打电话讨论。

### Step 0 — Pi 上的环境
```bash
i2cdetect -y 1               # 看到 0x68
ls /dev/ttyUSB*              # 看到 USB0/1/2/3 四个口
```

### Step 1 — IMU 单测（不上电机）
```bash
cd ~/qmini/sim2real
python3 -m deploy.tests.test_imu_gy91
# 手动倾斜机器人，肉眼检查上面 A 节的四个条件
```

### Step 2 — Mock 干跑（不接硬件）
```bash
python3 -m deploy.tests.test_mock_loop --onnx policy.onnx
# 看输出：50 Hz、ONNX <12ms、phase 增量 ≈ 0.0278
```

### Step 3 — 电机静态测试
机器人**架起来悬空**，跑：
```bash
python3 run_qmini.py --mock        # 先 mock 确认 ONNX 路径对
python3 -m deploy.tests.manual_motor_control   # 单关节手动发位置
```

### Step 4 — 实机站立（**机器人吊起或有人扶住**）
```bash
python3 run_qmini.py --constant-cmd --vx 0.0 --duration 5 --skip-imu-calib
# vx=0 时 policy 应输出接近 0 的残差，机器人原地小幅"踏步"但不前进
```

### Step 5 — 起步
```bash
python3 run_qmini.py --constant-cmd --vx 0.10 --duration 10
```

### Step 6 — 手柄接管
```bash
python3 run_qmini.py    # 默认 JoystickCommand
```

---

## 📋 已知风险/暂未处理

1. **`base_lin_vel = 0`**：训练时给了 ±0.1 m/s 噪声，理论上能撑住，但走快时可能不稳。后续要写一个 leg-odometry 状态估计器。
2. **gyro bias 不固化**：每次开机重新标定 3 秒。如果想固化，把打印出的 bias 填到 `config/calibration.yaml::imu.gyro_bias`，下次用 `--skip-imu-calib` 启动。
3. **磁力计/气压计未使用**：MPU9250 的 AK8963 磁力计和 BMP280 气压计当前没接入，对站立/低速行走不影响。
4. **急停只有软件**：`Ctrl+C` → `emergency_stop()` 把所有电机 kp/kd 置零。强烈建议加一个硬件急停按钮直接切电源。
5. **`run_qmini.py` 里 `_build_real()` 默认会尝试连手柄**，没插手柄时会打印 "no device found" 并返回零命令——程序不会崩，但要记得用 `--constant-cmd` 才能给非零速度。

---

## 📝 改动日志

- **2026-05-13** — 初版：补全 RealIMU + JoystickCommand，run_qmini.py 切到 real，写本 record.md
