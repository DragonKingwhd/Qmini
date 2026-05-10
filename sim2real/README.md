# Qmini Sim2Real 部署框架

把训练好的 Qmini ONNX 策略部署到树莓派 (Raspberry Pi) 上的实机。

```
速度命令 (vx, wz)            IMU (roll/pitch + gyro)        关节编码器 (10 joints)
        │                              │                              │
        ▼                              ▼                              ▼
            ┌──────────────────────────────────────────────────────┐
            │   ObservationBuilder  (45D × 3 history = 135D)        │
            └──────────────────────────────────────────────────────┘
                                       │
                                       ▼
                         ONNX Policy  (135 → 12)
                                       │
                                       ▼
            ┌──────────────────────────────────────────────────────┐
            │   BIRLPostProcessor:                                  │
            │     raw[12] -> [-1,1] clip -> scale to [LOW,HIGH]     │
            │     freq[2]    -> PhaseModulator.compute(freq)        │
            │     deltas[10] -> joint_target += deltas * dt         │
            └──────────────────────────────────────────────────────┘
                                       │
                                       ▼
                  关节位置目标  joint_target[10]  (rad)
                                       │
                                       ▼
                关节驱动器 (CAN/UART)  → 各电机位置闭环
```

控制频率 = `1 / (sim.dt × decimation) = 1 / (0.001 × 15) = 66.67 Hz`，与训练保持一致。

---

## 目录结构

```
sim2real/
├── README.md
├── run_qmini.py               真机启动脚本（默认 mock，TODO 改成真驱动）
├── config/
│   └── calibration.yaml       IMU 零偏等 per-robot 标定值
└── deploy/
    ├── __init__.py
    ├── constants.py           关节顺序 / 默认姿态 / 动作范围 / 频率
    ├── phase_modulator.py     训练侧 PhaseModulator 的 numpy 端口
    ├── observation.py         45D 单步 + 3 步历史的观测拼装
    ├── controller.py          ONNX 推理 + BIRL 动作后处理
    ├── calibration.py         IMU 零偏 / 初始姿态检查
    ├── main.py                66.7 Hz 主循环 QminiController
    ├── io/
    │   ├── __init__.py
    │   ├── interfaces.py      抽象驱动接口（IMU / Joint / CommandSource）
    │   └── mock.py            假驱动（无硬件端到端测试用）
    └── tests/
        ├── __init__.py
        └── test_mock_loop.py  Mock 循环测试
```

---

## 每个文件做什么

### `deploy/constants.py`
部署侧硬件常量的唯一来源。**必须**和训练侧严格一致：
- `JOINT_NAMES` —— 10 个关节的标准顺序（`hip_yaw_l, hip_roll_l, hip_pitch_l, knee_pitch_l, ankle_pitch_l, ` 然后 `_r` 一组）
- `DEFAULT_JOINT_POS` / `REF_JOINT_POS` —— 镜像 `qmini_env_cfg.py` 的初始姿态和 `BIRLActionTermCfg.ref_joint_pos`
- `INC_LOW / INC_HIGH = [0.5, 0.5] + [-15.0]*10 / [3.5, 3.5] + [+15.0]*10` —— 网络输出 `[-1,1]` 解扰到的范围（频率 + 关节速率）
- `OBS_PER_STEP = 49`, `OBS_HISTORY_LEN = 3`, `OBS_DIM = 135`
- `CONTROL_HZ = 66.67`, `CONTROL_DT ≈ 0.015 s`

### `deploy/io/interfaces.py`
三个抽象基类：
- `IMUDriver.read() -> (roll_rad, pitch_rad, ang_vel_xyz[3])`
- `JointDriver.read() -> (joint_pos[10], joint_vel[10])` + `send_position(target[10])`
- `CommandSource.read() -> (vx_m_s, wz_rad_s)`

其它代码全程**只**通过这些接口和硬件对话——你在树莓派上要写真实版本。

### `deploy/io/mock.py`
- `WigglingIMU` / `StaticIMU`：合成正弦或恒零的 IMU
- `MockJoints`：一阶动力学，模拟电机跟踪目标
- `ConstantCommand` / `WSCommand`：恒值或线程安全的速度命令源

### `deploy/phase_modulator.py`
训练侧 `BIRLActionTerm.PhaseModulator` 的 numpy CPU 单环境实现：
- 用网络输出的频率 `freq[2]` 驱动 `phase += 2π · freq · dt`
- 提供 `pm_phase`（4D：sin/cos）与 `pm_frequency_obs`（4D：`f*0.3-1` 重复）

部署侧的 phase 是**有状态**的，每步都要 `compute(freq)` 推进；如果 phase 不与训练一致，policy 给的步频/支撑相位会乱。

### `deploy/observation.py`
`ObservationBuilder` 把传感器读数拼成 45D，再级联最近 3 帧成 135D：
```
[ vx_cmd, wz_cmd,             (2)
  roll, pitch,                (2)
  ang_vel * 0.5,              (3)
  joint_pos - ref,            (10)
  joint_vel * 0.1,            (10)
  current_target - joint_pos, (10)   ← 注意：用上一步的 target
  pm_phase * static_flag,     (4)
  pm_freq  * static_flag ]    (4)
```
`static_flag = 1` 当 `‖[vx,0,wz]‖ ≥ 0.15`，否则 `0`（站立模式）。

### `deploy/controller.py`
- `ONNXPolicy`：CPU `onnxruntime` 推理，校验 in/out dims = 135/12
- `BIRLPostProcessor`：维护 `current_joint_target`，每步：
  1. `raw -> clip [-1,1] -> scale to [INC_LOW, INC_HIGH]`
  2. 拆分 `freq[:2]` 与 `joint_deltas[2:]`
  3. `current_joint_target += joint_deltas * step_dt`
  4. clamp 到关节软限位

### `deploy/calibration.py`
- `calibrate_imu_bias(...)`：开机时让机器人静止 3 秒，估 roll/pitch 零偏 + 陀螺仪 bias
- `check_initial_pose(...)`：开机检查关节是不是在 `DEFAULT_JOINT_POS` 附近
- `load_yaml_config(...)`：从 `config/calibration.yaml` 读已经标定的零偏

### `deploy/main.py`
`QminiController`：把上面所有模块串成 66.7 Hz 主循环。每步：
1. 读 IMU + 关节 + 命令
2. 用**上一步**的 joint target 算 obs（与训练一致）
3. ONNX 推理
4. 推理超时（>12 ms）或异常 → 保持上一步 target，跳过 phase update
5. 否则 `BIRLPostProcessor.step(raw)` → 同步推进 `PhaseModulator`
6. `joints.send_position(target)`

退出时（Ctrl+C 或 duration 到）调用 `joints.emergency_stop()`。

### `run_qmini.py`
真机启动脚本。当前导入 mock 驱动；写完 `deploy/io/real.py` 后改成真的。

### `deploy/tests/test_mock_loop.py`
不连硬件的端到端测试，验证：
1. 动作 shape 正确、有限
2. joint target 在限位内
3. 实际控制频率 ≈ 66.7 Hz
4. 推理延迟统计
5. 频率输出在 `[0.5, 3.5]`
6. 初始 target 等于默认姿态
7. 观测维度 = 135

---

## 部署到 Raspberry Pi

### 1. 拷贝文件

```bash
# 在开发机上
scp -r /home/user/Desktop/WHD/Qmini/sim2real pi@<ip>:/home/pi/qmini/
scp /home/user/Desktop/WHD/Qmini/Qmini/outputs/.../exported/policy.onnx pi@<ip>:/home/pi/qmini/sim2real/
```

### 2. 安装依赖（Pi 上）

```bash
pip3 install numpy pyyaml
pip3 install onnxruntime          # ARM 有 aarch64 wheel
# 你的 IMU / 电机库（pyserial、python-can、自家 SDK……）
```

### 3. 写真驱动 `deploy/io/real.py`

```python
import numpy as np
from .interfaces import IMUDriver, JointDriver, CommandSource

class RealIMU(IMUDriver):
    def __init__(self, port: str = "/dev/ttyUSB0"):
        # TODO: 打开串口/I2C 到你的 IMU
        ...
    def read(self) -> tuple[float, float, np.ndarray]:
        # TODO: 取一帧；roll/pitch 单位 rad；gyro 是 (3,) rad/s, body frame
        return roll, pitch, np.array([wx, wy, wz], dtype=np.float32)

class RealJoints(JointDriver):
    JOINT_HW_ORDER = [...]  # 你硬件链路上的 10 个 ID
    # 必须能映射回 deploy.constants.JOINT_NAMES 的顺序

    def __init__(self, can_iface: str = "can0"):
        # TODO: 打开 CAN / UART
        ...
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        # TODO: 读 10 路位置、速度（rad / rad·s⁻¹），按 JOINT_NAMES 顺序返回
        return joint_pos, joint_vel
    def send_position(self, target_rad: np.ndarray) -> None:
        # TODO: rad -> 你电机的指令单位（rev / count / deg）
        # 然后通过总线发出
        ...
    def emergency_stop(self) -> None:
        # TODO: 给所有电机发力矩=0 / 锁位
        ...

class JoystickCommand(CommandSource):
    def __init__(self, dev: str = "/dev/input/js0"):
        ...
    def read(self) -> tuple[float, float]:
        # TODO: 把摇杆映射到 vx ∈ [-0.3, 0.7], wz ∈ [-1.0, 1.0]
        return vx, wz
```

> ⚠️ **关节顺序**是最容易出错的地方。`deploy.constants.JOINT_NAMES` 定义了**唯一**正典顺序，
> 你的硬件总线 ID 排序必须在 `RealJoints` 内部做映射；Isaac Lab 的 `Articulation`
> 也按 URDF 顺序，已和 `JOINT_NAMES` 一致。

### 4. 标定 IMU 零偏（每次开机或机械动过）

启动时 `run_qmini.py` 会自动跑 3 秒静态采集，结果只保存在内存。如果你想固化下来，把
打印出的 `bias_rp` / `bias_gyro` 填进 `config/calibration.yaml` 然后用 `--skip-imu-calib`。

### 5. 启动

```bash
cd /home/pi/qmini/sim2real
python3 run_qmini.py --onnx policy.onnx --vx 0.0 --wz 0.0 --duration 5
```

先用 `vx=0, wz=0` 跑几秒看机器人能不能稳定站立——`static_flag=0` 时 phase
信号被乘 0，policy 应该输出接近 0 的关节增量。再逐步加速。

---

## 部署前 checklist

| 项 | 检查办法 |
|---|---|
| ☐ Mock 测试通过 | `python -m deploy.tests.test_mock_loop --onnx policy.onnx` 能过 |
| ☐ IMU 单位是弧度 | 90° 倾倒时 `pitch ≈ ±1.57`；不是的话在 `RealIMU.read` 转 |
| ☐ IMU 轴向对齐 base | 机器人前倾，`pitch` 变化（不是 roll/yaw） |
| ☐ 关节顺序 | `joints.read()` 返回的 `pos[0]` 真的是 `hip_yaw_l` |
| ☐ 关节方向对齐训练 | `pos` 和训练侧 `joint_pos` 同符号——给一个关节强制写 +0.1 rad，URDF 仿真侧应该往同方向走 |
| ☐ 关节零位 = URDF 零位 | 不一致就在 `RealJoints` 里加 offset |
| ☐ 软限位填了 | `constants.py:JOINT_LIMIT_LOW/HIGH` 不是 `None` |
| ☐ ONNX 在 Pi 上能 < 12 ms | 看 `mock_loop` 输出的 `inference: max ... ms` |

---

## 训练 ↔ 部署对齐检查

改训练侧任一硬件参数都必须**同步**改部署侧：

| 训练侧文件/字段 | 部署侧对应 |
|---|---|
| `qmini_env_cfg.py:joint_pos` (init) | `deploy/constants.py:DEFAULT_JOINT_POS` |
| `mdp/actions.py:BIRLActionTermCfg.ref_joint_pos` | `deploy/constants.py:REF_JOINT_POS` |
| `mdp/actions.py:BIRLActionTermCfg.inc_low/high_ranges` | `deploy/constants.py:INC_LOW/INC_HIGH` |
| `qmini_env_cfg.py:decimation`, `sim.dt` | `deploy/constants.py:CONTROL_HZ` |
| `mdp/actions.py:_convert_phi` | `deploy/constants.py:CONVERT_PHI` |
| `mdp/observations.py` 各 scale/clamp | `deploy/observation.py` 同步 |
| `ObservationsCfg.PolicyCfg.history_length` | `deploy/constants.py:OBS_HISTORY_LEN` |

---

## 常见问题

### 推理慢（>12 ms on Pi）
- 树莓派 4B 的 ARM CPU 跑 135→12 MLP 一般在 1-3 ms 之间；如果到 20 ms+，先检查是不是用了 wrong wheel
- 若必须降速，把 `CONTROL_HZ` 改成 `33.33`（同时改训练侧 decimation）然后**重训**——直接降部署频率会让 phase modulator 跑不准

### 机器人一启动就抽搐
最常见的原因是**关节顺序**或**符号**搞错。建议先把 `RealJoints.send_position` 临时改成
直接发 `DEFAULT_JOINT_POS`，看机器人是不是平静地立着。

### 站着不动也飘
检查 `static_flag` 在 `vx=wz=0` 时是不是 `0`——是的话 phase 信号会被乘 0，policy
应该输出接近 0 的关节增量；如果不是，看 `STATIC_CMD_NORM_THRESHOLD` 和训练侧
`_get_static_flag` 阈值是否一致。

### Phase 越走越快
`PhaseModulator.compute` 必须**每步都调**，且 `dt` 严格等于训练时的 `env.step_dt`。
如果实际 loop 落到 50 Hz，phase 推进只走了应有的 75%，policy 给的支撑/摆动相位就错了。

---

## 紧急停机

主循环中按 **Ctrl+C**：
- 退出主循环
- `finally` 调 `joints.emergency_stop()`（你在 `RealJoints` 里实现成"力矩=0 + 锁位"或类似）

**更安全的做法**是硬件上接一个独立急停按钮，直接切电机驱动器电源——软件急停只是最后一道防线。
