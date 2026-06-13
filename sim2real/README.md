# Qmini Sim2Real 部署框架

把训练好的 Qmini ONNX 策略部署到树莓派 (Raspberry Pi) 上的实机。

```
速度命令 [vx, vy, wz]      IMU (lin_vel/ang_vel/proj_g)     关节编码器 (10 joints)
        │                              │                              │
        ▼                              ▼                              ▼
            ┌──────────────────────────────────────────────────────┐
            │  ObservationBuilder  (44D, 单步, 无历史)              │
            └──────────────────────────────────────────────────────┘
                                       │
                                       ▼
                         ONNX Policy  (44 → 10)
                                       │
                                       ▼
            ┌──────────────────────────────────────────────────────┐
            │   GaitActionPostProcessor:                            │
            │     1. raw -> clip [-1, 1]                            │
            │     2. ReferenceGait.advance()                        │
            │     3. offsets  = ref_gait.offsets()                  │
            │     4. target   = default + offsets + raw * 0.10      │
            │     5. clamp 到关节软限位                              │
            └──────────────────────────────────────────────────────┘
                                       │
                                       ▼
                  关节位置目标  joint_target[10]  (rad)
                                       │
                                       ▼
                关节驱动器 (CAN/UART)  → 各电机位置闭环
```

控制频率 = `1 / (sim.dt × decimation) = 1 / (0.005 × 4) = 50 Hz`，与训练保持一致。

---

## 每次开机流程（先看这个）

GO-M8010-6 的 q 是相对上电位置的 → **每次断电零位都作废**，开机按这四步走：

```bash
cd ~/Desktop/Qmini
# 1. 机器人放台架,两腿伸直竖直、左右平行、脚尖朝前、脚掌放平,上电
python3 sim2real/power_on.py                # 2. 一条龙:重标零位+低增益验证,全✅才放行
python3 sim2real/run_qmini.py --keyboard    # 3. 不断电,直接跑策略(键盘控制)
```

(`power_on.py` = `calib/stand_zero.py` + `debug/goto_default.py` 串联;
 只是 git pull / Pi 重启而电机没断电 → 两步都不用,直接跑 run_qmini。)

第 4 步默认有**确认门**:ramp 到 DEFAULT + IMU 标定后,电机全刚度锁住等你回车——
此时把机器人从台架拿下来放地面扶稳,回车策略才接管(跳过加 `--no-wait`)。
键盘: `w/s`=加减速(0~0.3 m/s) `a/d`=左/右转(±0.5 rad/s) `空格`=原地踏步。
训练命令分布 vx∈[0.1,0.3]、vy≡0、wz∈[±0.5],别指望它后退或侧移。

sign / 软限位 / PD 增益不随断电丢失，只有 `motor_zero_rad` 需要每次重标。
首次标定或机械动过 → 用 `calib/capture_zero.py` 双限位法（见 `calib/readme.md`）。

---

## 目录结构

```
sim2real/
├── README.md
├── run_qmini.py               真机启动入口(--mock 桌面空跑)
├── power_on.py                电机上电后一条龙:重标零位+验证(必跑)
├── policy.onnx                导出的策略
├── config/
│   └── calibration.yaml       per-robot 标定值(motor_zero/sign/gyro bias)
├── deploy/                    运行时核心包(只放主循环用到的代码)
│   ├── constants.py           关节顺序 / 默认姿态 / 步态参数 / 软限位 / 频率
│   ├── reference_gait.py      训练侧 QminiReferenceGaitAction 的 numpy 端口
│   ├── observation.py         44D 单步观测拼装
│   ├── controller.py          ONNX 推理 + 残差动作后处理
│   ├── calibration.py         IMU gyro bias / 初始姿态检查
│   ├── main.py                50 Hz 主循环 QminiController
│   └── io/
│       ├── interfaces.py      抽象驱动接口(IMU / Joint / CommandSource)
│       ├── mock.py            假驱动(无硬件端到端测试用)
│       ├── real.py            真驱动(GO-M8010-6 ×10 / GY-91 IMU / 手柄)
│       └── keyboard.py        键盘速度命令源(--keyboard, w/s/a/d/空格)
├── calib/                     标定工作流 → 见 calib/readme.md
│   ├── stand_zero.py          ★每次开机:台架直腿一键零位
│   ├── capture_zero.py        首次/机械动过:双限位法解 sign+zero
│   ├── calibrate_sign.py      逐关节看运动方向验 sign
│   └── fix_hip_roll_zero.py   只补 hip_roll 零位(腿竖直=joint0)
├── debug/                     硬件调试/单项测试 → 见 debug/readme.md
├── docs/
│   └── record.md              测试记录
└── logs/                      运行日志
```

---

## 每个文件做什么

### `deploy/constants.py`
部署侧硬件常量的唯一来源。**必须**和训练侧严格一致：
- `JOINT_NAMES` —— 10 个关节的标准顺序（hip→knee→ankle，左腿后右腿）
- `DEFAULT_JOINT_POS` —— 镜像 `qmini_env_cfg.py:init_state.joint_pos`，
  既是残差动作的"零点"，也是 `joint_pos_rel` 观测中减去的偏移
- 步态参数 (`GAIT_PERIOD_S=0.72`, `GAIT_STANCE_RATIO=0.60`,
  `HIP_PITCH_AMPLITUDE=0.22`, `KNEE_PITCH_AMPLITUDE=0.24`,
  `ANKLE_PITCH_AMPLITUDE=0.14`, `PUSH_OFF_ANKLE_SCALE=0.18`)
  —— 必须等于 `qmini_env_cfg.py:ActionsCfg.joint_pos` 里的字段
- `ACTION_SCALE = 0.10` —— 残差动作放大系数
- `OBS_DIM = 44`, `ACTION_DIM = 10`
- `CONTROL_HZ = 50.0`

### `deploy/reference_gait.py`
训练侧 `QminiReferenceGaitAction` 的 **numpy 单环境实现**：
- 维护标量 `phase ∈ [0, 1)`，每步 `phase += step_dt / gait_period`
- 左腿用 `phase`，右腿用 `(phase + 0.5) % 1`（反相位）
- `offsets()` 用分段 stance/swing 函数算出 `(hip, knee, ankle)` 的参考偏移，
  右腿三个 pitch 关节符号取反（默认姿态左右镜像）
- `phase_obs` 提供 `[sin(2π·phase), cos(2π·phase)]`，喂进观测的 `gait_phase`

部署侧的 phase 是**有状态**的；如果跑不到 50 Hz 真实频率，实际推进的相位就和
训练时不一致，policy 出来的动作会和参考步态错位。

### `deploy/io/interfaces.py`
三个抽象基类：
- `IMUDriver.read() -> (lin_vel_b[3], ang_vel_b[3], proj_g_b[3])` —— **body frame**
- `JointDriver.read() -> (joint_pos[10], joint_vel[10])` + `send_position(target[10])`
- `CommandSource.read() -> [vx, vy, wz]`

⚠️ **`base_lin_vel` 是 IMU 上不直接可观测的量**——训练时用的是仿真里的真值。
真机上你有两条路：
1. 写一个状态估计器（IMU + 腿部 odometry 互补/Kalman 融合出 body-frame 速度）
2. 直接给 0，依赖训练时这一通道上加的噪声做 sim2real margin
默认把这件事委托给 `RealIMU` 自己处理。

### `deploy/io/mock.py`
- `WigglingIMU` / `StaticIMU`：合成 IMU（gravity 用小角度近似投影到 body frame）
- `MockJoints`：一阶动力学，模拟电机跟踪目标
- `ConstantCommand` / `WSCommand`：恒值或线程安全的速度命令

### `deploy/observation.py`
`ObservationBuilder` 把传感器读数拼成 44D（**没有历史**，直接喂网络）：
```
[ base_lin_vel_b,        (3)
  base_ang_vel_b,        (3)
  projected_gravity_b,   (3)
  velocity_command,      (3)   [vx, vy, wz]
  joint_pos - default,   (10)
  joint_vel,             (10)
  last_action,           (10)  ← 上一步 ONNX 的 raw 输出
  [sin(2π·phase), cos(2π·phase)] ]  (2)
```

**每一项都在 body frame 里**——驱动里别给世界坐标系的量。

### `deploy/controller.py`
- `ONNXPolicy`：CPU `onnxruntime`，校验 in/out dims = 44/10
- `GaitActionPostProcessor`：每步：
  1. `raw -> clip [-1, 1]`
  2. `ref_gait.advance()` 推进相位一个 `step_dt`
  3. `offsets = ref_gait.offsets()`
  4. `target = default_pos + offsets + raw * 0.10`
  5. clamp 到关节软限位

### `deploy/calibration.py`
- `calibrate_imu_gyro(...)` —— 静止 3 秒采 gyro 平均值作为 bias
- `check_initial_pose(...)` —— 开机检查关节是否在 `DEFAULT_JOINT_POS` 附近
- `load_yaml_config(...)` —— 从 `config/calibration.yaml` 读已固化的标定值

### `deploy/main.py`
`QminiController`：把上面所有模块串成 50 Hz 主循环。每步：
1. 读 IMU + 关节 + 速度命令（关节读数减 `joint_offset`）
2. 用**当前**的 `gait.phase_obs` 算 obs（与训练一致：obs 拿到的是上一动作步推进过的相位）
3. ONNX 推理
4. 推理超时（>12 ms）或异常 → 保持上一步 target，但 phase 仍要 advance
5. 否则 `GaitActionPostProcessor.step(raw)` —— 内部已 advance phase
6. `joints.send_position(target + joint_offset)`

退出时（Ctrl+C 或 duration 到）调用 `joints.emergency_stop()`。

### `run_qmini.py`
真机启动脚本。默认用 `deploy/io/real.py` 真驱动，`--mock` 切换到假驱动桌面空跑。
找不到 `config/calibration.yaml` 会直接退出（无标定 = sign+1/zero=0 = 顶死电机）。

### `debug/test_mock_loop.py`
不连硬件的端到端测试，验证：
1. 动作 shape 正确、有限
2. joint target 在限位内
3. 实际控制频率 ≈ 50 Hz
4. 推理延迟统计
5. gait phase 单调推进且每步增量 ≈ `dt/period = 0.02/0.72 ≈ 0.0278`
6. 观测维度 = 44

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
        # TODO: 打开串口/I2C 到你的 IMU；可能还要起一个状态估计线程
        ...
    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # body-frame: 线速度 m/s, 角速度 rad/s, 重力单位向量
        return lin_vel_b, ang_vel_b, proj_gravity_b

class RealJoints(JointDriver):
    JOINT_HW_ORDER = [...]  # 你硬件链路上的 10 个 ID
    # 必须能映射回 deploy.constants.JOINT_NAMES 的顺序

    def __init__(self, can_iface: str = "can0"):
        # TODO: 打开 CAN / UART
        ...
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        # 按 JOINT_NAMES 顺序返回 (pos, vel)
        return joint_pos, joint_vel
    def send_position(self, target_rad: np.ndarray) -> None:
        # rad -> 你电机的指令单位（rev / count / deg），按 JOINT_NAMES 取
        ...
    def emergency_stop(self) -> None:
        ...

class JoystickCommand(CommandSource):
    def read(self) -> np.ndarray:
        # 摇杆映射到 [vx, vy, wz]
        return np.array([vx, vy, wz], dtype=np.float32)
```

> ⚠️ **关节顺序**是最容易出错的地方。`deploy.constants.JOINT_NAMES` 是**唯一**正典顺序，
> 你的硬件总线 ID 排序必须在 `RealJoints` 内部做映射。同时确认训练时
> Isaac Lab Articulation 内部的 joint 顺序也是这个——可以加一行 print
> 在训练侧 `actions.py` 里 `print(self._joint_names)` 验证。

### 4. 标定（每次开机或机械动过）

**关节零位**：每次上电先跑 `calib/stand_zero.py`（见顶部"每次开机流程"和
`calib/readme.md`）。机械动过或怀疑零位 → `calib/capture_zero.py` 双限位法重标。

**IMU**：启动时 `run_qmini.py` 会自动跑 3 秒静态 gyro 标定，bias 留在内存。
如果你想固化：把打印出的 bias 填进 `config/calibration.yaml` 的
`imu.gyro_bias`，再用 `--skip-imu-calib` 启动跳过开机标定。

### 5. 启动

```bash
cd /home/pi/qmini/sim2real
python3 run_qmini.py --onnx policy.onnx --vx 0.0 --duration 5
```

先用 `vx=0` 跑几秒看机器人能不能稳定站立——参考步态的偏移依然会让腿
做小幅摆动，但身体应该不会前进。再用 `--vx 0.10`（与训练 curriculum 起步速度
一致）正式起步。

---

## 部署前 checklist

| 项 | 检查办法 |
|---|---|
| ☐ Mock 测试通过 | `python3 debug/test_mock_loop.py --onnx policy.onnx` 能过 |
| ☐ IMU 单位 = SI | 角速度 rad/s（不是 deg/s），线速度 m/s |
| ☐ IMU 在 body frame | 机器人前倾时 `proj_g[0]` 变正、`ang_vel[1]` 负→正 |
| ☐ 关节顺序 | `joints.read()` 返回的 `pos[0]` 真的是 `hip_yaw_l` |
| ☐ 关节方向 | 给一个关节强制写 +0.1 rad，URDF 仿真侧应该往同方向走 |
| ☐ 关节零位 = URDF 零位 | 不一致就在 `calibration.yaml:joints.offset` 里写补偿 |
| ☐ 软限位填了 | `constants.py:JOINT_LIMIT_LOW/HIGH` 不是 `None`（强烈建议） |
| ☐ ONNX 在 Pi 上 < 12 ms | 看 `mock_loop` 输出的 `inference: max ... ms` |
| ☐ 默认姿态对得上 | `check_initial_pose` 没报警告 |

---

## 训练 ↔ 部署对齐检查

改训练侧任一硬件参数都必须**同步**改部署侧：

| 训练侧文件/字段 | 部署侧对应 |
|---|---|
| `qmini_env_cfg.py:init_state.joint_pos` | `deploy/constants.py:DEFAULT_JOINT_POS` |
| `qmini_env_cfg.py:ActionsCfg.joint_pos.scale` | `deploy/constants.py:ACTION_SCALE` |
| `qmini_env_cfg.py:ActionsCfg.joint_pos.gait_period` | `deploy/constants.py:GAIT_PERIOD_S` |
| `qmini_env_cfg.py:ActionsCfg.joint_pos.stance_ratio` | `deploy/constants.py:GAIT_STANCE_RATIO` |
| `…hip/knee/ankle_pitch_amplitude` | `deploy/constants.py:*_AMPLITUDE` |
| `…push_off_ankle_scale` | `deploy/constants.py:PUSH_OFF_ANKLE_SCALE` |
| `qmini_env_cfg.py:decimation`, `sim.dt` | `deploy/constants.py:CONTROL_HZ` |
| `mdp/observations.py` 各 obs term | `deploy/observation.py` 各字段及顺序 |

---

## 常见问题

### 推理慢（>12 ms on Pi）
- 树莓派 4B 的 ARM CPU 跑 44→10 这种小 MLP 一般在 1-3 ms 之间；如果到 20 ms+，
  先检查是不是用了 wrong wheel（用 aarch64 的 onnxruntime）
- 若必须降速，把 `CONTROL_HZ` 改成 25 Hz（同时改训练侧 decimation）然后**重训**——
  直接降部署频率会让 reference gait 推进过慢

### 机器人一启动就抽搐
最常见的原因是**关节顺序**或**符号**搞错。建议先把 `RealJoints.send_position`
临时改成直接发 `DEFAULT_JOINT_POS_VEC`，看机器人是不是平静地立着。然后再发
`default + ref_gait.offsets()`，看腿有没有协调地踏步。最后再让 policy 接管。

### 站着不动也飘
- `velocity_command = [0, 0, 0]` 时 policy 应输出接近 0 的残差
- 检查 `last_action` 是不是真的反馈了上一步 raw_action（`set_last_action` 每步都要调）
- 检查 IMU `proj_g` 是不是有大 bias

### 真实速度估计 (`base_lin_vel`)
真机上很难直接拿到。如果你没接状态估计器：
- 临时做法：让 `RealIMU.read()` 返回 `np.zeros(3)` 当 lin_vel
- 训练时 `Unoise(n_min=-0.1, n_max=0.1)` 给了一定容忍，但偏差越大策略越不稳
- 建议在 Pi 上跑一个简单的腿部前向运动学估计：站立脚为参考点，根据关节角速度
  反推 base 平动速度

### 步态相位漂移
`ReferenceGait.advance()` 必须**每步都调**，且实际控制周期严格 ≈ 0.02 s。
监控 `mock_loop` 输出的 phase 增量，应该接近 `0.02 / 0.72 ≈ 0.0278`。
若你的实际 dt 抖动大（比如串口阻塞了几个 ms），考虑用本地时钟测量真实 dt
来动态推进 phase 而不是死乘 `step_dt`。

---

## 紧急停机

主循环按 **Ctrl+C**：
- 退出主循环
- `finally` 调 `joints.emergency_stop()`（你在 `RealJoints` 里实现成"力矩=0 + 锁位"或类似）

**更安全的做法**是硬件上接独立急停按钮，直接切电机驱动器电源——软件急停只是最后一道防线。
