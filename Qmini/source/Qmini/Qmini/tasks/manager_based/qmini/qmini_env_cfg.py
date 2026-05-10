"""Velocity walking configuration for the Qmini biped."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp


_QMINI_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "assets"
)
_QMINI_USD_PATH = os.path.join(_QMINI_ASSETS_DIR, "Qmini_ref.usd")


@configclass
class QminiWalkSceneCfg(InteractiveSceneCfg):
    """Flat terrain scene for Qmini walking."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_QMINI_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.45),
            joint_pos={
                "hip_yaw_l": 0.4,
                "hip_roll_l": -0.1,
                "hip_pitch_l": -1.5,
                "knee_pitch_l": 1.0,
                "ankle_pitch_l": -1.3,
                "hip_yaw_r": -0.4,
                "hip_roll_r": 0.1,
                "hip_pitch_r": 1.5,
                "knee_pitch_r": -1.0,
                "ankle_pitch_r": 1.3,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness={
                    "hip_yaw.*": 55.0,
                    "hip_roll.*": 105.0,
                    "hip_pitch.*": 75.0,
                    "knee.*": 45.0,
                    "ankle.*": 30.0,
                },
                damping={
                    "hip_yaw.*": 0.3,
                    "hip_roll.*": 2.5,
                    "hip_pitch.*": 0.3,
                    "knee.*": 0.5,
                    "ankle.*": 0.25,
                },
                effort_limit={
                    "hip_yaw.*": 20.0,
                    "hip_roll.*": 60.0,
                    "hip_pitch.*": 20.0,
                    "knee.*": 20.0,
                    "ankle.*": 20.0,
                },
            ),
        },
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.0,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )


@configclass
class CommandsCfg:
    """Forward-only velocity commands for initial walking training."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.10, 0.10),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Reference gait plus residual joint position actions."""

    joint_pos = mdp.QminiReferenceGaitActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.10,
        gait_period=0.72,
        stance_ratio=0.60,
        hip_pitch_amplitude=0.22,
        knee_pitch_amplitude=0.24,
        ankle_pitch_amplitude=0.14,
        push_off_ankle_scale=0.18,
    )


@configclass
class ObservationsCfg:
    """Policy observations for velocity walking."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase_obs)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Domain randomization and resets."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.3, 0.3),
            "operation": "add",
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.4, 0.4)},
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.2, 0.2)}},
    )


@configclass
class RewardsCfg:
    """Walking rewards adapted from the Guguji biped velocity task."""

    track_lin_vel_x = RewTerm(
        func=mdp.track_lin_vel_x_yaw_frame_exp,
        weight=4.8,
        params={"command_name": "base_velocity", "std": 0.10},
    )
    forward_progress = RewTerm(func=mdp.forward_progress_yaw_frame, weight=4.0)
    alive_bonus = RewTerm(func=mdp.is_alive, weight=0.6)

    upright = RewTerm(func=mdp.upright_reward, weight=1.6)
    height = RewTerm(func=mdp.height_reward, weight=1.0, params={"target_height": 0.45})

    hip_alternation = RewTerm(
        func=mdp.hip_alternation_reward,
        weight=1.6,
        params={
            "left_hip_name": "hip_pitch_l",
            "right_hip_name": "hip_pitch_r",
            "target_separation": 0.42,
            "antiphase_sigma": 0.25,
        },
    )
    knee_flexion = RewTerm(
        func=mdp.knee_flexion_reward,
        weight=0.6,
        params={"left_knee_name": "knee_pitch_l", "right_knee_name": "knee_pitch_r", "target": 1.0, "sigma": 0.35},
    )
    knee_symmetry = RewTerm(
        func=mdp.knee_symmetry_penalty,
        weight=-0.5,
        params={"left_knee_name": "knee_pitch_l", "right_knee_name": "knee_pitch_r"},
    )
    hip_symmetry = RewTerm(
        func=mdp.hip_symmetry_penalty,
        weight=-0.5,
        params={"left_hip_name": "hip_pitch_l", "right_hip_name": "hip_pitch_r"},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["ankle_pitch_l", "ankle_pitch_r"]),
            "command_name": "base_velocity",
            "threshold": 0.20,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    lateral_velocity = RewTerm(func=mdp.lin_vel_y_l2, weight=-0.3)
    yaw_rate = RewTerm(func=mdp.ang_vel_z_l2, weight=-0.4)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5)
    backward_velocity = RewTerm(func=mdp.backward_velocity_penalty_yaw_frame, weight=-2.8)
    stall_penalty = RewTerm(
        func=mdp.stall_penalty_yaw_frame,
        weight=-4.0,
        params={"command_name": "base_velocity", "threshold": 0.08},
    )
    undesired_knee_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["knee_pitch_l", "knee_pitch_r"]), "threshold": 1.0},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["ankle_pitch_l", "ankle_pitch_r"])},
    )


@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.9})
    base_height = DoneTerm(func=mdp.base_height_below_threshold, params={"minimum_height": 0.20})


@configclass
class CurriculumCfg:
    """Forward velocity curriculum."""

    velocity_command = CurrTerm(
        func=mdp.velocity_command_curriculum,
        params={"command_name": "base_velocity", "min_vel": 0.10, "max_vel": 0.30, "success_threshold": 0.8},
    )


@configclass
class QminiWalkEnvCfg(ManagerBasedRLEnvCfg):
    """Qmini flat-terrain walking task."""

    scene: QminiWalkSceneCfg = QminiWalkSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class QminiWalkEnvCfg_PLAY(QminiWalkEnvCfg):
    """Evaluation configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
