"""Qmini deploy-side constants.

Mirror of the training side — values MUST match
source/Qmini/Qmini/tasks/manager_based/qmini/qmini_env_cfg.py and mdp/actions.py.
If you change anything here, update training side and re-export the policy.
"""

from __future__ import annotations

import math

# ---- Joint order (matches Isaac Lab articulation order, which follows URDF) ----
# IMPORTANT: when wiring real motors, this is the canonical index used inside
# the policy. The real driver must permute its hardware-side ordering to match.
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
NUM_LEGS = 2

# ---- Default standing pose (= ArticulationCfg.init_state.joint_pos) ----
DEFAULT_JOINT_POS: list[float] = [0.4, -0.1, -1.5, 1.0, -1.3, -0.4, 0.1, 1.5, -1.0, 1.3]

# ---- Reference pose used in observations (BIRLActionTermCfg.ref_joint_pos) ----
REF_JOINT_POS: list[float] = [0.4, -0.1, -1.5, 1.0, -1.3, -0.4, 0.1, 1.5, -1.0, 1.3]

# ---- BIRL action scaling: 12 = 2 phase frequencies + 10 joint deltas (rad/s) ----
ACTION_DIM = NUM_LEGS + NUM_JOINTS  # 12
INC_LOW: list[float]  = [0.5, 0.5] + [-15.0] * 10
INC_HIGH: list[float] = [3.5, 3.5] + [+15.0] * 10
ACTION_CLIP = 1.0  # rsl_rl wrapper clip_actions

# ---- Phase modulator ----
CONVERT_PHI = 1.2 * math.pi  # phase < CONVERT_PHI -> support, else swing

# ---- Observation scaling factors (must mirror mdp/observations.py) ----
ANG_VEL_SCALE = 0.5
JOINT_VEL_SCALE = 0.1

# ---- Observation history ----
OBS_PER_STEP = 45           # see mdp/observations.py (PolicyCfg): 2+2+3+10+10+10+4+4
OBS_HISTORY_LEN = 3
OBS_DIM = OBS_PER_STEP * OBS_HISTORY_LEN  # 135

# ---- Static-flag threshold (mdp/observations.py::_get_static_flag) ----
STATIC_CMD_NORM_THRESHOLD = 0.15

# ---- Joint soft limits ----
# Use the URDF soft limits at deploy time as an extra safety clamp.
# Values left as None mean "trust the URDF / firmware-side limit". Override
# with measured values once you've calibrated mechanical hard stops.
JOINT_LIMIT_LOW:  list[float | None] = [None] * 10
JOINT_LIMIT_HIGH: list[float | None] = [None] * 10

# ---- Control loop ----
# Training: sim.dt = 0.001, decimation = 15  -> step_dt = 0.015 s -> 66.67 Hz.
# Keep deploy at the same rate so phase integration matches.
CONTROL_HZ = 66.6667
CONTROL_DT = 1.0 / CONTROL_HZ  # ~0.015 s

# ---- Safety thresholds ----
INFERENCE_TIMEOUT_S = 0.012  # hold previous target if inference > 12 ms
ACTION_RATE_LIMIT = 50.0     # rad/s, slew limit on commanded joint targets


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
