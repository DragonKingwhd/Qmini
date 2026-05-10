# record.md — Qmini Walk 双足行走训练系统

## 项目配置和操作指南

### 1. 安装项目 / 检查环境

```bash
# 激活 isaaclab 环境
conda activate isaaclab

# 安装项目（editable mode）
cd /home/a/Desktop/github/Qmini/Qmini
pip install -e source/Qmini

# 列出可用环境（确认 Qmini-Walk-v0 已注册）
python scripts/list_envs.py

# 运行随机动作测试（验证仿真能启动）
python3 scripts/random_agent.py --task=Qmini-Walk-v0 --num_envs 1

# 运行零动作测试（验证机器人能站立）
python scripts/zero_agent.py --task=Qmini-Walk-v0 --num_envs 1
```

### 2. 开始训练

```bash
cd /home/a/Desktop/github/Qmini/Qmini

# 基础训练（headless，4096 环境，5000 iterations）
python scripts/rsl_rl/train.py --task=Qmini-Walk-v0 --headless

# 自定义参数训练
python scripts/rsl_rl/train.py --task=Qmini-Walk-v0 --num_envs 4096 --max_iterations 5000 --headless

# 带可视化训练（调试用，环境数建议减少）
python scripts/rsl_rl/train.py --task=Qmini-Walk-v0 --num_envs 64
```

### 3. 继续上次训练

```bash
# 从最新 checkpoint 继续
python scripts/rsl_rl/train.py --task=Qmini-Walk-v0 --resume --headless

# 从指定 checkpoint 继续
python scripts/rsl_rl/train.py --task=Qmini-Walk-v0 --resume \
    --load_run 2026-04-04_xx-xx-xx --load_checkpoint model_2000.pt \
    --max_iterations 5000 --headless
```

### 4. 演示训练结果

```bash
# 播放最新模型（可视化）
python scripts/rsl_rl/play.py --task=Qmini-Walk-Play-v0 --num_envs 64

# 从指定 checkpoint 播放
python scripts/rsl_rl/play.py --task=Qmini-Walk-Play-v0 --num_envs 1 \
    --load_run 2026-04-04_xx-xx-xx --load_checkpoint model_5000.pt

# 录制视频
python scripts/rsl_rl/play.py --task=Qmini-Walk-Play-v0 --num_envs 1 \
    --video --video_length 600 --headless
```

### 5. 导出 ONNX（部署到真机）

```bash
# 导出最优 checkpoint 为 ONNX
python scripts/rsl_rl/play.py --task=Qmini-Walk-Play-v0 --num_envs 1 \
    --load_run <run_name> --load_checkpoint model_5000.pt \
    --export_onnx --headless
```

### 6. TensorBoard 监控

```bash
# 查看所有训练
tensorboard --logdir logs/rsl_rl/qmini_walk --port 6006

# 查看特定训练
tensorboard --logdir logs/rsl_rl/qmini_walk/<timestamp> --port 6006
```

---

## 系统架构

### 与 RoboTamer4Qmini（原始 Isaac Gym 版）的区别

| 维度 | RoboTamer4Qmini (Isaac Gym) | Qmini Walk (本项目, Isaac Lab) |
|------|----------------------------|-------------------------------|
| 仿真框架 | Isaac Gym (legged_gym) | Isaac Lab (ManagerBasedRLEnv) |
| 环境基类 | 自定义 LeggedRobotEnv | QminiWalkEnv (ManagerBasedRLEnv 子类) |
| 配置方式 | Python 类继承 (Base/BIRL) | @configclass 装饰器 (声明式) |
| 动作系统 | 自定义 BIRLTask.action() | QminiReferenceGaitAction (ActionTerm 子类) |
| 观测系统 | 手动 torch.cat + deque 历史 | ObsTerm + history_length=3 (内置) |
| 奖励系统 | 单函数返回 rew_dict | 29 个独立 RewTerm + 共享状态 |
| 命令系统 | 自定义 _resample_commands | UniformVelocityCommandCfg (内置) |
| 域随机化 | 手动在 step 中实现 | EventTermCfg (startup/reset/interval) |
| 传感器延迟 | DelayDeque (10-50 步) | v1 暂未移植（后续可添加） |
| 训练框架 | 自定义 PPO | RSL-RL (标准 Isaac Lab 集成) |
| Gym 注册 | 无 | gymnasium.register("Qmini-Walk-v0") |

### 观测空间

#### Actor 观测（49 维 × 3 历史 = 147 维）

| 分量 | 维度 | 缩放 | 说明 |
|------|------|------|------|
| 速度命令 (vx, yaw_rate) | 2 | — | command_manager |
| 基座欧拉角 (roll, pitch) | 2 | ×1.0 | euler_xyz_from_quat |
| 基座角速度 (body frame) | 3 | ×0.5 | root_ang_vel_b |
| 关节位置偏差 (pos - ref) | 10 | — | joint_pos - ref_joint_pos |
| 关节速度 | 10 | ×0.1 | joint_vel |
| 关节位置误差 (target - actual) | 10 | — | current_target - joint_pos |
| 相位信号 sin/cos | 4 | — | sin(phase), cos(phase) × 2 腿 |
| 相位频率信号 | 4 | — | (freq×0.3 - 1) × 2 腿 |
| **static_flag 掩码** | — | — | 命令为零时相位/频率信号归零 |

#### 历史机制
- Isaac Lab 内置 `ObservationGroupCfg.history_length = 3`
- 自动维护 3 帧滑动窗口，flatten 后拼接
- 总维度: 49 × 3 = **147 维**

#### Critic 观测（特权信息 × 3 历史）

在 Actor 观测基础上增加：
- 线速度跟踪误差 (1D)、角速度跟踪误差 (1D)
- 基座线速度 (3D)
- 动作目标偏差 (10D)
- 最后网络输出 (10D × 2)
- 足部高度 (2D)、基座高度 (1D)
- 足部速度 (6D)、基座加速度 (3D)
- 足部接触力 (2D)
- 无延迟版本的 euler/ang_vel/joint_pos/joint_vel/joint_err (35D)

### 动作空间（12 维）

| 索引 | 类型 | 范围 (缩放后) | 说明 |
|------|------|-------------|------|
| [0] | 相位频率（左腿） | [0.5, 3.5] Hz | 控制左腿摆动频率 |
| [1] | 相位频率（右腿） | [0.5, 3.5] Hz | 控制右腿摆动频率 |
| [2-11] | 关节位置增量 | [-15, 15] rad/s | 增量模式: target += delta × dt |

**增量动作处理流程**:
1. 网络输出 [-1, 1]^12 → 缩放到动作范围
2. 前 2 维更新相位调制器频率
3. 后 10 维作为关节位置增量: `target += delta × step_dt`
4. 裁剪到 URDF 关节限位
5. 发送到 ImplicitActuator (PD 控制器)

### 相位调制器（BIRL 核心）

```
PhaseModulator:
  phase(t+1) = (phase(t) + 2π × frequency × dt) mod 2π
  
  支撑相: phase ∈ [0, 1.2π]     → foot_support_mask = True
  摆动相: phase ∈ (1.2π, 2π)    → foot_swing_mask = True
  
  观测信号: [sin(phase_L), sin(phase_R), cos(phase_L), cos(phase_R)]
  
  反相协调: 理想状态下两腿相位差 π（一腿支撑时另一腿摆动）
```

### 命令空间（3 维）

| 分量 | 范围 | 说明 |
|------|------|------|
| vx | [-0.3, 0.7] m/s | 前进/后退 |
| vy | 0 m/s | 不控制侧移 |
| yaw_rate | [-1.0, 1.0] rad/s | 旋转 |

命令每 5 秒随机重新采样。2% 的环境命令为零（站立）。

### 奖励函数（29 项）

#### 主要任务奖励

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| constant | +0.3 | 存活奖励 |
| base_height | +1.0 | exp(-70×(z-0.45)²)，维持 0.45m 站立高度 |
| balance | +1.5 | 高度 × 姿态的综合平衡奖励 |
| forward_velocity | +2.3 | exp 核跟踪前进速度命令 |
| yaw_rate | +2.5 | exp 核跟踪偏航角速度命令 |
| lateral_velocity | +0.7 | 惩罚侧向漂移 |
| vertical_velocity | +0.6 | 惩罚垂直弹跳 |
| angular_velocity | +0.6 | 惩罚 roll/pitch 角速度 |
| twist | +2.5 | 惩罚 roll/pitch 角度偏差 |

#### 步态质量奖励

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| foot_clearance | +1.0 | 摆动腿离地（力<1N） |
| foot_support | +0.7 | 支撑腿着地（力≥10N） |
| foot_height | +0.7 | 摆动腿抬脚高度 (0-5cm) |
| foot_soft_contact | ×2.7×balance | 平滑着地（力变化率） |
| feet_contact_force | +0.001 | 接触力分布合理性 |
| foot_slip | ×0.5×balance | 惩罚支撑腿滑动 |
| foot_vertical_velocity | ×0.2×balance | 惩罚脚部下落速度 |
| foot_acceleration | ×0.05×balance | 惩罚脚部加速度 |
| foot_phase_coordination | ×0.3×balance | 双足反相步态协调 |

#### 动作平滑性奖励

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| action_smoothness | ×1.5×balance | 关节位置 jerk 惩罚 |
| net_out_smoothness | ×0.001×balance | 网络输出 jerk 惩罚 |
| action_constraint | ×0.2×balance | 偏离参考姿态惩罚 |
| support_ankle_constraint | ×0.1×balance | 支撑腿关节固定惩罚 |

#### 关节惩罚

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| joint_pos_error | ×0.2×balance | 关节位置跟踪误差 |
| joint_velocity | ×0.003×balance | 关节速度惩罚 |
| joint_torque | +0.001 | 超出力矩限制惩罚 |

#### 其他

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| base_acceleration | ×0.1×balance | 基座加速度惩罚 |
| leg_width | ×0.5×balance | 维持 0.14m 步宽 |
| foot_pitch | ×0.5×balance | 惩罚脚部俯仰角 |
| phase_freq | ×0.03×balance | 相位频率平滑性 |
| net_out_value | ×0.0001×balance | 网络输出正则化 |

**注意**：许多奖励项会乘以 `balance_rew` 作为门控——当机器人姿态不稳定时自动降低权重，优先恢复平衡。总奖励裁剪到 `min=0`。

### 终止条件

| 条件 | 阈值 |
|------|------|
| 超时 | episode_length_s = 10s |
| 翻倒 | \|roll\| > 0.7 rad 或 \|pitch\| > 0.7 rad |
| 掉落 | z < 0.2 m |
| 基座碰撞 | base_link 接触力 > 1N |

### 域随机化

| 事件 | 模式 | 参数 |
|------|------|------|
| 摩擦系数 | startup | static/dynamic: [0.2, 1.5] |
| 质量缩放 | startup | ×[0.5, 1.5] 所有刚体 |
| 根状态随机 | reset | xy: ±0.5m, yaw: ±0.2 rad |
| 关节重置 | reset | 默认位置 |
| PD 增益缩放 | reset | stiffness/damping: ×[0.8, 1.2] |
| 随机推力 | interval 3s | xy 速度: ±0.5 m/s |
| 观测噪声 | ObsTerm.noise | euler ±0.15, ang_vel ±0.15, joint_pos ±0.05, joint_vel ±0.06 |

### 网络架构

```
Actor 观测 (147D) ─→ MLP: 147 → 512 → 256 → 12 (ReLU) ─→ 动作
Critic 观测 (特权) ─→ MLP: ?D → 512 → 256 → 1  (ReLU) ─→ 价值
```

### PPO 超参数

| 参数 | 值 |
|------|-----|
| num_envs | 4096 |
| num_steps_per_env | 24 |
| max_iterations | 5000 |
| learning_rate | 1e-3 |
| gamma | 0.995 |
| lambda (GAE) | 0.95 |
| clip_param | 0.2 |
| entropy_coef | 0.0005 |
| desired_kl | 0.01 |
| num_learning_epochs | 3 |
| num_mini_batches | 4 |
| schedule | adaptive |
| empirical_normalization | False |
| init_noise_std | 0.8 |

### 物理参数

| 参数 | 值 |
|------|-----|
| sim dt | 0.001 s (1000 Hz) |
| decimation | 15 |
| control freq | ~66.7 Hz |
| step_dt | 0.015 s |
| episode length | 10 s (~667 步) |

### 机器人参数

#### 关节配置（10 DOF）

| 关节名 | 类型 | 限位 (rad) | effort (Nm) | Kp | Kd |
|--------|------|-----------|-------------|-----|------|
| hip_yaw_l | revolute | [-0.1, 0.7] | 20 | 55.0 | 0.3 |
| hip_roll_l | revolute | [-0.3, 0.6] | 60 | 105.0 | 2.5 |
| hip_pitch_l | revolute | [-2.1, 0.0] | 20 | 75.0 | 0.3 |
| knee_pitch_l | revolute | [0.0, 2.1] | 20 | 45.0 | 0.5 |
| ankle_pitch_l | revolute | [-2.5, 0.0] | 20 | 30.0 | 0.25 |
| hip_yaw_r | revolute | [-0.7, 0.1] | 20 | 55.0 | 0.3 |
| hip_roll_r | revolute | [-0.6, 0.3] | 60 | 105.0 | 2.5 |
| hip_pitch_r | revolute | [0.0, 2.1] | 20 | 75.0 | 0.3 |
| knee_pitch_r | revolute | [-2.1, 0.0] | 20 | 45.0 | 0.5 |
| ankle_pitch_r | revolute | [0.0, 2.5] | 20 | 30.0 | 0.25 |

#### 默认站立姿态

```python
hip_yaw_l=0.4, hip_roll_l=-0.1, hip_pitch_l=-1.5, knee_pitch_l=1.0, ankle_pitch_l=-1.3
hip_yaw_r=-0.4, hip_roll_r=0.1, hip_pitch_r=1.5, knee_pitch_r=-1.0, ankle_pitch_r=1.3
```

初始高度: 0.45m，质量: 4.78 kg (base_link)

---

## 文件结构

```
~/Desktop/Qmini/
├── record.md                              ← 本文件
└── Qmini/
    ├── scripts/
    │   ├── list_envs.py
    │   ├── random_agent.py
    │   ├── zero_agent.py
    │   └── rsl_rl/
    │       ├── cli_args.py
    │       ├── train.py                   # 训练脚本（标准模板）
    │       └── play.py                    # 播放/导出脚本（标准模板）
    └── source/Qmini/
        ├── setup.py
        └── Qmini/
            ├── __init__.py
            ├── assets/
            │   └── q1/
            │       ├── urdf/q1.urdf       # Qmini 机器人 URDF ★
            │       └── meshes/            # 11 个 STL 碰撞/视觉网格 ★
            │           ├── base_link.STL
            │           ├── hip_yaw_l/r.STL
            │           ├── hip_roll_l/r.STL
            │           ├── hip_pitch_l/r.STL
            │           ├── knee_pitch_l/r.STL
            │           └── ankle_pitch_l/r.STL
            └── tasks/manager_based/qmini/
                ├── __init__.py            # gym.register ★
                ├── qmini_env_cfg.py       # 环境配置（场景+机器人+MDP全部） ★
                ├── qmini_env.py           # QminiWalkEnv 自定义环境类 ★
                ├── agents/
                │   └── rsl_rl_ppo_cfg.py  # PPO 训练参数 ★
                └── mdp/
                    ├── __init__.py        # MDP 导出 ★
                    ├── actions.py         # 参考步态 + residual 动作 ★
                    ├── observations.py    # velocity walking 观测函数 ★
                    ├── rewards.py         # 行走速度/姿态/步态奖励 ★
                    ├── terminations.py    # 终止条件 ★
                    └── events.py          # 域随机化事件 ★
```

（★ = 本次新建/重写的文件）

---

## 变更记录

### 2026-05-10 新建 Qmini-Walk velocity walking 任务

**目标**：
- 删除旧的 `Qmini-BIRL-v0` 注册，改为 `Qmini-Walk-v0` / `Qmini-Walk-Play-v0`
- 参考 `guguji_isaaclab` 的双足行走任务，使用“参考步态 + policy residual”的动作形式
- 奖励重点改为前向速度跟踪、身体直立、base 高度、左右腿反相、单腿支撑、少滑脚

**创建/重写的文件**：
1. `assets/Qmini_ref.usd` — 引用版机器人 USD 资产
2. `mdp/actions.py` — **QminiReferenceGaitAction**（10 维关节 residual 动作）
3. `mdp/observations.py` — gait phase 观测补充，其他观测复用 Isaac Lab 内置项
4. `mdp/rewards.py` — velocity walking 奖励函数
5. `mdp/terminations.py` — twist_over + height_over
6. `mdp/curriculums.py` — 前向速度课程
7. `mdp/__init__.py` — 重新导出所有 MDP 模块
8. `qmini_env.py` — **QminiWalkEnv**
9. `qmini_env_cfg.py` — 完整环境配置（场景/机器人/命令/观测/奖励/终止/事件）
10. `agents/rsl_rl_ppo_cfg.py` — PPO 超参数（[512,256,128] MLP, γ=0.99, 5000 iter）
11. `__init__.py` — 注册 Qmini-Walk-v0 和 Qmini-Walk-Play-v0

**参考来源**：
- `~/Desktop/github/guguji_isaaclab/source/guguji_locomotion/.../locomotion/velocity/`
- Isaac Lab 内置 locomotion MDP 项

**关键设计决策**：
- 用参考步态给双足相位先验，策略只学习 residual 修正
- 初始命令只训练前向速度，课程从 0.10 m/s 增到 0.30 m/s
- 先用平地训练，粗糙地形/传感器延迟后续再加

**状态**：✅ 代码完成，任务注册验证通过

---

## 待办事项

- [x] 生成引用版 USD 资产
- [x] 实现 QminiReferenceGaitAction
- [x] 实现 velocity walking 观测/奖励函数
- [x] 实现终止条件
- [x] 配置域随机化
- [x] 编写环境配置和自定义环境类
- [x] 配置 PPO 超参数
- [x] 注册 Gymnasium 环境
- [ ] 安装项目 (`pip install -e .`)
- [ ] 运行零动作测试（验证机器人站立）
- [ ] 运行随机动作测试（验证动作管线）
- [ ] 开始训练 (5000 iterations)
- [ ] 训练收敛后导出 ONNX
- [ ] 添加传感器延迟（v2 优化）
- [ ] 添加粗糙地形课程（v2 优化）
