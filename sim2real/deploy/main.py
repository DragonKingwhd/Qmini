"""50 Hz Qmini deploy control loop.

Wires together: IMU + joints + velocity-command drivers, ONNX policy,
reference gait, observation builder. Hardware drivers are passed in by
the caller so the same loop runs with mocks (offline test) or real
drivers on the Raspberry Pi.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibration import Calibration, calibrate_imu_gyro, check_initial_pose, load_yaml_config
from .constants import (
    ACTION_DIM,
    CONTROL_DT,
    DEFAULT_JOINT_POS_VEC,
    INFERENCE_TIMEOUT_S,
    NUM_JOINTS,
)
from .controller import GaitActionPostProcessor, ONNXPolicy
from .io.interfaces import CommandSource, IMUDriver, JointDriver
from .observation import ObservationBuilder
from .reference_gait import ReferenceGait


@dataclass
class StepRecord:
    t: float
    cmd: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_lin_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    proj_g: np.ndarray = field(default_factory=lambda: np.zeros(3))
    joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))
    joint_target: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))
    raw_action: np.ndarray = field(default_factory=lambda: np.zeros(ACTION_DIM))
    gait_phase: float = 0.0
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
        # ONNX export from rsl_rl prepends Sub(mean)/Div(std) — feed RAW obs.
        self.policy = ONNXPolicy(onnx_path)
        self.obs_builder = ObservationBuilder()
        self.gait = ReferenceGait()
        self.post = GaitActionPostProcessor(self.gait)

        self.imu = imu
        self.joints = joints
        self.cmd_source = cmd_source

        self._calib = Calibration()
        if calibration_yaml is not None:
            cfg = load_yaml_config(calibration_yaml)
            imu_cfg = cfg.get("imu", {})
            if imu_cfg.get("gyro_bias") is not None:
                self._calib.imu_gyro_bias = np.asarray(imu_cfg["gyro_bias"], dtype=np.float32)
            joints_cfg = cfg.get("joints", {})
            if joints_cfg.get("offset") is not None:
                offset = np.asarray(joints_cfg["offset"], dtype=np.float32)
                if offset.shape == (NUM_JOINTS,):
                    self._calib.joint_offset = offset

        self._last_target = np.asarray(DEFAULT_JOINT_POS_VEC, dtype=np.float32).copy()
        self._history: list[StepRecord] | None = [] if record_history else None
        self._running = False

    # ---- setup ----
    def calibrate_imu(self, duration_s: float = 3.0) -> None:
        self._calib.imu_gyro_bias = calibrate_imu_gyro(self.imu, duration_s=duration_s)

    def check_pose(self, tol_rad: float = 0.15) -> None:
        check_initial_pose(self.joints, tol_rad=tol_rad)

    # ---- per-step ----
    def step(self) -> StepRecord:
        # 1. read sensors / command
        lin_vel, ang_vel, proj_g = self.imu.read()
        ang_vel = np.asarray(ang_vel, dtype=np.float32) - self._calib.imu_gyro_bias

        joint_pos, joint_vel = self.joints.read()
        joint_pos = np.asarray(joint_pos, dtype=np.float32) - self._calib.joint_offset
        joint_vel = np.asarray(joint_vel, dtype=np.float32)

        cmd = np.asarray(self.cmd_source.read(), dtype=np.float32).reshape(3)

        # 2. build observation. Uses gait phase from BEFORE this step's
        #    advance, matching training: in qmini_env_cfg.py the obs term
        #    `gait_phase_obs` reads `action_term.gait_phase`, which is the
        #    phase produced by the previous action step.
        obs = self.obs_builder.build(
            base_lin_vel_b=lin_vel,
            base_ang_vel_b=ang_vel,
            projected_gravity_b=proj_g,
            velocity_command=cmd,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            gait_phase_sin_cos=self.gait.phase_obs,
        )

        # 3. inference
        try:
            raw_action, infer_s = self.policy.infer(obs)
            if infer_s > INFERENCE_TIMEOUT_S:
                print(f"[WARN] inference {infer_s * 1000:.1f} ms over budget; holding previous target")
                target = self._last_target
                # Still advance the gait so we don't drift further out of phase.
                self.gait.advance()
            else:
                target = self.post.step(raw_action)
        except Exception as e:
            print(f"[ERROR] inference failed: {e!r}; holding previous target")
            target = self._last_target
            raw_action = np.zeros(ACTION_DIM, dtype=np.float32)
            self.gait.advance()
            infer_s = 0.0

        # 4. send to robot
        self.joints.send_position(target + self._calib.joint_offset)
        self.obs_builder.set_last_action(raw_action)
        self._last_target = np.asarray(target, dtype=np.float32).copy()

        rec = StepRecord(
            t=time.perf_counter(),
            cmd=cmd.copy(),
            base_lin_vel=np.asarray(lin_vel).copy(),
            base_ang_vel=np.asarray(ang_vel).copy(),
            proj_g=np.asarray(proj_g).copy(),
            joint_pos=joint_pos.copy(),
            joint_target=np.asarray(target).copy(),
            raw_action=np.asarray(raw_action, dtype=np.float32).copy(),
            gait_phase=float(self.gait.phase),
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
                sleep_s = CONTROL_DT - (time.perf_counter() - t_loop)
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
