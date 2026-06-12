# debug — 硬件调试 / 单项测试

| 脚本 | 用途 | 需要硬件 |
|---|---|---|
| `goto_default.py` | 纯"去 DEFAULT 并保持"，无策略/IMU/ONNX。低 kp（默认 0.4×）隔离验证电机路径，标定后必跑 | ✓ |
| `test_mock_loop.py` | 假驱动端到端跑控制回路，验 obs 维度/限位/50Hz/推理延迟。`python3 debug/test_mock_loop.py --onnx policy.onnx` | ✗ |
| `manual_motor_control.py` | 键盘逐电机点动（j/k 小步、h/l 大步、f 卸力、p 锁位） | ✓ |
| `identify_motor.py` | 单电机轴向+减速比识别（电机端动 1 rad 目测关节转角） | ✓ |
| `identify_axes_manual.py` | 手扳关节识别轴向（零力矩，看 q 变化方向） | ✓ |
| `pid_sweep.py` | kp/kd 扫参找安全增益。已得结论：kp=1.20 kd=0.10；**kd≥0.5 必振荡过流** | ✓ |
| `test_imu_gy91.py` | GY-91（MPU9250）I2C 读数自检 | ✓ |

带硬件的脚本都默认机器人**悬空/台架**，异常立刻 Ctrl+C（都会卸力矩退出）。
