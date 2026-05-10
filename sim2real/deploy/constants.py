"""Qmini deploy-side constants.

Mirror of the training side — values MUST match
``source/Qmini/Qmini/tasks/manager_based/qmini/qmini_env_cfg.py`` and
``mdp/actions.py``. If you change anything here, update the training side
and re-export the policy.
"""

from __future__ import annotations

# ---- Joint order ----
# Canonical joint order used by both the policy and the deploy package.
# Isaac Lab's Articulation orders joints by USD prim order (which follows
# the original URDF order for this asset). The hardware driver MUST permute
# its bus-side ordering to match this list.
JOINT_NAMES: list[str] = [
    "hip_yaw_l",
    "hip_roll_l",
    "hip_pitch_l",
    "knee_pitch_l",
    "ankle_pitch_l",
    "hip_yaw_r",
    "hip_roll_r",
    "hip_pitch_r",
    "knee_pitch_r",
    "ankle_pitch_r",
]
NUM_JOINTS = len(JOINT_NAMES)  # 10

# ---- Default joint positions ----
# Mirrors qmini_env_cfg.py:QminiWalkSceneCfg.robot.init_state.joint_pos.
# Used both as the "zero" of the residual action and as the offset
# subtracted in joint_pos_rel observation.
DEFAULT_JOINT_POS: dict[str, float] = {
    "hip_yaw_l":     0.4,
    "hip_roll_l":   -0.1,
    "hip_pitch_l":  -1.5,
    "knee_pitch_l":  1.0,
    "ankle_pitch_l":-1.3,
    "hip_yaw_r":    -0.4,
    "hip_roll_r":    0.1,
    "hip_pitch_r":   1.5,
    "knee_pitch_r": -1.0,
    "ankle_pitch_r": 1.3,
}
DEFAULT_JOINT_POS_VEC: list[float] = [DEFAULT_JOINT_POS[n] for n in JOINT_NAMES]

# ---- Reference gait parameters (= QminiReferenceGaitActionCfg defaults) ----
GAIT_PERIOD_S = 0.72
GAIT_STANCE_RATIO = 0.60
HIP_PITCH_AMPLITUDE = 0.22
KNEE_PITCH_AMPLITUDE = 0.24
ANKLE_PITCH_AMPLITUDE = 0.14
PUSH_OFF_ANKLE_SCALE = 0.18

# ---- Residual action scale ----
# Mirrors QminiReferenceGaitActionCfg.scale (0.10).
ACTION_SCALE = 0.10
ACTION_DIM = NUM_JOINTS  # 10
ACTION_CLIP = 1.0        # rsl_rl wrapper default

# ---- Observation dims (single-step, no history) ----
# Order is fixed by qmini_env_cfg.py:ObservationsCfg.PolicyCfg:
#   base_lin_vel       (3)   body frame, m/s
#   base_ang_vel       (3)   body frame, rad/s
#   projected_gravity  (3)   gravity vector projected into body frame
#   velocity_commands  (3)   [lin_vel_x_cmd, lin_vel_y_cmd, ang_vel_z_cmd]
#   joint_pos_rel     (10)   joint_pos - default_joint_pos
#   joint_vel_rel     (10)   joint_vel - default_joint_vel  (default_vel = 0)
#   last_action       (10)   raw policy output of the previous step
#   gait_phase_obs     (2)   [sin(2π·phase), cos(2π·phase)]
OBS_DIM = 3 + 3 + 3 + 3 + 10 + 10 + 10 + 2  # 44

# ---- Joint soft limits (rad) ----
# Used as a final safety clamp on the commanded joint target. ``None`` means
# "trust whatever the firmware enforces". Override with measured values
# once mechanical hard stops are characterised.
JOINT_LIMIT_LOW:  list[float | None] = [None] * 10
JOINT_LIMIT_HIGH: list[float | None] = [None] * 10

# ---- Control loop ----
# Training: sim.dt=0.005, decimation=4 -> step_dt = 0.020 s -> 50 Hz.
# Keep deploy at the same rate so the reference-gait phase advances at the
# same speed as during training.
CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ  # 0.02 s

# ---- Safety thresholds ----
INFERENCE_TIMEOUT_S = 0.012  # hold previous target if inference > 12 ms


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
