"""66.7 Hz Qmini deploy control loop.

Wires together: IMU + joints + velocity-command drivers, ONNX policy,
phase modulator, observation builder. Hardware drivers are passed in by
the caller so the same loop runs with mocks (offline test) or real
drivers on the Raspberry Pi.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibration import Calibration, calibrate_imu_bias, check_initial_pose, load_yaml_config
from .constants import (
    CONTROL_DT,
    INFERENCE_TIMEOUT_S,
    NUM_JOINTS,
    NUM_LEGS,
)
from .controller import BIRLPostProcessor, ONNXPolicy
from .io.interfaces import CommandSource, IMUDriver, JointDriver
from .observation import ObservationBuilder
from .phase_modulator import PhaseModulator


@dataclass
class StepRecord:
    t: float
    vx_cmd: float
    wz_cmd: float
    roll: float
    pitch: float
    joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))
    joint_target: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))
    raw_action: np.ndarray = field(default_factory=lambda: np.zeros(NUM_LEGS + NUM_JOINTS))
    freq: np.ndarray = field(default_factory=lambda: np.zeros(NUM_LEGS))
    inference_s: float = 0.0


class QminiController:
    def __init__(
        self,
        onnx_path: str | Path,
        imu: IMUDriver,
        joints: JointDriver,
        cmd_source: CommandSource,
        calibration_yaml: str | Path | None = None,
        record_history: bool = False,
    ) -> None:
        # ONNX is exported with rsl_rl's prepended Sub(mean)/Div(std), so feed RAW obs.
        self.policy = ONNXPolicy(onnx_path)
        self.obs_builder = ObservationBuilder()
        self.post = BIRLPostProcessor()
        self.phase = PhaseModulator(time_step=CONTROL_DT)

        self.imu = imu
        self.joints = joints
        self.cmd_source = cmd_source

        self._calib = Calibration()
        if calibration_yaml is not None:
            cfg = load_yaml_config(calibration_yaml)
            imu_cfg = cfg.get("imu", {})
            if imu_cfg.get("bias_rp") is not None:
                br, bp = imu_cfg["bias_rp"]
                self._calib.imu_bias_rp = (float(br), float(bp))
            if imu_cfg.get("bias_gyro") is not None:
                self._calib.imu_bias_gyro = np.asarray(imu_cfg["bias_gyro"], dtype=np.float32)

        self._last_target = np.asarray(self.post.current_joint_target, dtype=np.float32).copy()
        self._history: list[StepRecord] | None = [] if record_history else None
        self._running = False

    # ---- setup ----
    def calibrate_imu(self, duration_s: float = 3.0) -> None:
        bias_rp, bias_gyro = calibrate_imu_bias(self.imu, duration_s=duration_s)
        self._calib.imu_bias_rp = bias_rp
        self._calib.imu_bias_gyro = bias_gyro

    def check_pose(self, tol_rad: float = 0.15) -> None:
        check_initial_pose(self.joints, tol_rad=tol_rad)

    # ---- per-step ----
    def step(self) -> StepRecord:
        # 1. read sensors / command
        roll_raw, pitch_raw, gyro_raw = self.imu.read()
        roll = roll_raw - self._calib.imu_bias_rp[0]
        pitch = pitch_raw - self._calib.imu_bias_rp[1]
        ang_vel = np.asarray(gyro_raw, dtype=np.float32) - self._calib.imu_bias_gyro

        joint_pos, joint_vel = self.joints.read()
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        joint_vel = np.asarray(joint_vel, dtype=np.float32)

        vx_cmd, wz_cmd = self.cmd_source.read()

        # 2. build observation (uses *previous* joint target, like training)
        obs = self.obs_builder.build(
            vx_cmd=vx_cmd, wz_cmd=wz_cmd,
            roll=roll, pitch=pitch,
            ang_vel_xyz=ang_vel,
            joint_pos=joint_pos, joint_vel=joint_vel,
            current_joint_target=self.post.current_joint_target,
            pm_phase_sig=self.phase.pm_phase,
            pm_freq_sig=self.phase.pm_frequency_obs,
        )

        # 3. inference
        try:
            raw_action, infer_s = self.policy.infer(obs)
            if infer_s > INFERENCE_TIMEOUT_S:
                print(f"[WARN] inference {infer_s * 1000:.1f} ms over budget; holding previous target")
                target = self._last_target
                freq = self.phase.frequency  # keep last
            else:
                # Order matters and mirrors training:
                #   process_actions advances the phase WITH the new freq AND
                #   integrates joint deltas in the same step.
                freq, target = self.post.step(raw_action)
                self.phase.compute(freq)
        except Exception as e:
            print(f"[ERROR] inference failed: {e!r}; holding previous target")
            target = self._last_target
            raw_action = np.zeros(NUM_LEGS + NUM_JOINTS, dtype=np.float32)
            freq = self.phase.frequency
            infer_s = 0.0

        # 4. send to robot
        self.joints.send_position(target)
        self._last_target = np.asarray(target, dtype=np.float32).copy()

        rec = StepRecord(
            t=time.perf_counter(),
            vx_cmd=float(vx_cmd), wz_cmd=float(wz_cmd),
            roll=float(roll), pitch=float(pitch),
            joint_pos=joint_pos.copy(), joint_target=np.asarray(target).copy(),
            raw_action=np.asarray(raw_action, dtype=np.float32).copy(),
            freq=np.asarray(freq, dtype=np.float32).copy(),
            inference_s=infer_s,
        )
        if self._history is not None:
            self._history.append(rec)
        return rec

    # ---- main loop ----
    def run(self, duration_s: float | None = None) -> None:
        self._running = True
        t_start = time.perf_counter()
        n_steps = 0
        try:
            while self._running:
                t_loop = time.perf_counter()
                if duration_s is not None and t_loop - t_start >= duration_s:
                    break
                self.step()
                n_steps += 1
                # Loose pacing: sleep dt - elapsed. Real-time biped control on
                # a Pi is ARM CPU-bound; if we miss deadline, we miss it.
                dt_used = time.perf_counter() - t_loop
                sleep_s = CONTROL_DT - dt_used
                if sleep_s > 0:
                    time.sleep(sleep_s)
        finally:
            self._running = False
            try:
                self.joints.emergency_stop()
            except Exception as e:
                print(f"[WARN] emergency_stop failed: {e!r}")
            elapsed = time.perf_counter() - t_start
            rate = n_steps / elapsed if elapsed > 0 else 0.0
            print(f"[INFO] stopped: {n_steps} steps in {elapsed:.1f} s ({rate:.1f} Hz)")

    def stop(self) -> None:
        self._running = False

    @property
    def history(self) -> list[StepRecord] | None:
        return self._history
