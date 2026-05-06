"""Environment configuration for Qmini BIRL bipedal locomotion task."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from .mdp.actions import BIRLActionTermCfg

##
# Asset path
##

# __file__ is at .../tasks/manager_based/qmini/qmini_env_cfg.py
# assets are at .../Qmini/assets/ — need to go up 4 levels (qmini -> manager_based -> tasks -> Qmini)
_QMINI_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "assets"
)
_QMINI_URDF_PATH = os.path.join(_QMINI_ASSETS_DIR, "q1", "urdf", "q1.urdf")


##
# Scene definition
##


@configclass
class QminiBIRLSceneCfg(InteractiveSceneCfg):
    """Configuration for the Qmini BIRL locomotion scene."""

    # ground terrain (flat plane for v1)
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

    # Qmini bipedal robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=_QMINI_URDF_PATH,
            fix_base=False,
            merge_fixed_joints=True,
            replace_cylinders_with_capsules=True,
            activate_contact_sensors=True,
            joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
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

    # contact sensor on foot links
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.0,  # updated at physics dt
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for velocity tracking."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.7),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications: BIRL action term (12-dim: 2 freq + 10 joints)."""

    birl_action = BIRLActionTermCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    """Observation specifications for actor and critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations: 49D per step x 3 history = 147D total."""

        velocity_commands = ObsTerm(func=mdp.velocity_commands_xz)
        base_euler = ObsTerm(
            func=mdp.base_euler_rp,
            noise=Unoise(n_min=-0.15, n_max=0.15),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_scaled,
            noise=Unoise(n_min=-0.15, n_max=0.15),
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel_to_ref,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_scaled,
            noise=Unoise(n_min=-0.06, n_max=0.06),
        )
        joint_pos_err = ObsTerm(func=mdp.joint_pos_error)
        phase_sig = ObsTerm(func=mdp.phase_signal)
        phase_freq = ObsTerm(func=mdp.phase_freq_signal)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 3
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations: privileged info with history."""

        velocity_commands = ObsTerm(func=mdp.velocity_commands_xz)
        cmd_lin_vel_err = ObsTerm(func=mdp.cmd_lin_vel_error)
        cmd_ang_vel_err = ObsTerm(func=mdp.cmd_ang_vel_error)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_euler = ObsTerm(func=mdp.base_euler_rp_privileged)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel_scaled_privileged)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel_to_ref_privileged)
        joint_vel = ObsTerm(func=mdp.joint_vel_scaled_privileged)
        action_target_rel = ObsTerm(func=mdp.action_target_rel_to_ref)
        joint_pos_err = ObsTerm(func=mdp.joint_pos_error_privileged)
        phase_sig = ObsTerm(func=mdp.phase_signal)
        phase_freq = ObsTerm(func=mdp.phase_freq_signal)
        last_net_out = ObsTerm(func=mdp.last_net_out_joints)
        foot_height = ObsTerm(func=mdp.foot_height_obs)
        base_height = ObsTerm(func=mdp.base_height_obs)
        foot_vel = ObsTerm(func=mdp.foot_vel_obs)
        base_acc = ObsTerm(func=mdp.base_acc_obs)
        foot_force = ObsTerm(
            func=mdp.foot_force_obs,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["ankle_pitch_l", "ankle_pitch_r"])},
        )
        # Duplicate last_net_out (matches source)
        last_net_out_2 = ObsTerm(func=mdp.last_net_out_joints)
        # Delayed actor obs (same as actor but without noise in critic)
        base_euler_delayed = ObsTerm(func=mdp.base_euler_rp)
        base_ang_vel_delayed = ObsTerm(func=mdp.base_ang_vel_scaled)
        joint_pos_rel_delayed = ObsTerm(func=mdp.joint_pos_rel_to_ref)
        joint_vel_delayed = ObsTerm(func=mdp.joint_vel_scaled)
        joint_pos_err_delayed = ObsTerm(func=mdp.joint_pos_error)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 3
            self.flatten_history_dim = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Domain randomization events."""

    # Startup events
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.5),
            "dynamic_friction_range": (0.2, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )

    # Reset events
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.2, 0.2)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # Interval events
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 3.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class RewardsCfg:
    """All 29 BIRL reward terms."""

    # Primary rewards
    constant = RewTerm(func=mdp.constant_reward, weight=0.3)
    base_height = RewTerm(func=mdp.base_height_reward, weight=1.0)
    balance = RewTerm(func=mdp.balance_reward, weight=1.5)
    forward_velocity = RewTerm(func=mdp.forward_velocity_reward, weight=2.3)
    yaw_rate = RewTerm(func=mdp.yaw_rate_reward, weight=2.5)
    lateral_velocity = RewTerm(func=mdp.lateral_velocity_reward, weight=0.7)
    vertical_velocity = RewTerm(func=mdp.vertical_velocity_reward, weight=0.6)
    angular_velocity = RewTerm(func=mdp.angular_velocity_reward, weight=0.6)
    twist = RewTerm(func=mdp.twist_reward, weight=2.5)

    # Acceleration reward (weight includes balance_rew multiplication inside func)
    base_acceleration = RewTerm(func=mdp.base_acceleration_reward, weight=0.1)

    # Foot rewards
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["ankle_pitch_l", "ankle_pitch_r"])},
    )
    foot_support = RewTerm(
        func=mdp.foot_support_reward,
        weight=0.7,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["ankle_pitch_l", "ankle_pitch_r"])},
    )
    foot_height = RewTerm(
        func=mdp.foot_height_reward,
        weight=0.7,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["ankle_pitch_l", "ankle_pitch_r"])},
    )
    foot_soft_contact = RewTerm(
        func=mdp.foot_soft_contact_reward,
        weight=2.7,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["ankle_pitch_l", "ankle_pitch_r"])},
    )
    feet_contact_force = RewTerm(
        func=mdp.feet_contact_force_reward,
        weight=0.001,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["ankle_pitch_l", "ankle_pitch_r"])},
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip_reward,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["ankle_pitch_l", "ankle_pitch_r"])},
    )
    foot_vertical_velocity = RewTerm(func=mdp.foot_vertical_velocity_reward, weight=0.2)
    foot_acceleration = RewTerm(func=mdp.foot_acceleration_reward, weight=0.05)

    # Posture rewards
    leg_width = RewTerm(func=mdp.leg_width_reward, weight=0.5)
    foot_pitch = RewTerm(func=mdp.foot_pitch_reward, weight=0.5)

    # Smoothness rewards
    action_smoothness = RewTerm(func=mdp.action_smoothness_reward, weight=1.5)
    net_out_smoothness = RewTerm(func=mdp.net_out_smoothness_reward, weight=0.001)

    # Constraint rewards
    action_constraint = RewTerm(func=mdp.action_constraint_reward, weight=0.2)
    support_ankle_constraint = RewTerm(
        func=mdp.support_ankle_constraint_reward,
        weight=0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["ankle_pitch_l", "ankle_pitch_r"])},
    )
    joint_pos_error = RewTerm(func=mdp.joint_pos_error_reward, weight=0.2)

    # Energy rewards
    joint_velocity = RewTerm(func=mdp.joint_velocity_reward, weight=0.003)
    joint_torque = RewTerm(func=mdp.joint_torque_reward, weight=0.001)

    # Regularization rewards
    phase_freq = RewTerm(func=mdp.phase_freq_reward, weight=0.03)
    net_out_value = RewTerm(func=mdp.net_out_value_reward, weight=0.0001)

    # Gait coordination
    foot_phase_coordination = RewTerm(func=mdp.foot_phase_coordination_reward, weight=0.3)


@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    twist_over = DoneTerm(func=mdp.twist_over, params={"max_angle": 0.7})
    height_over = DoneTerm(func=mdp.height_over, params={"min_height": 0.2})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]),
            "threshold": 1.0,
        },
    )


##
# Environment configuration
##


@configclass
class QminiBIRLEnvCfg(ManagerBasedRLEnvCfg):
    """Complete configuration for Qmini BIRL locomotion environment."""

    # Scene settings
    scene: QminiBIRLSceneCfg = QminiBIRLSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings: dt=0.001, decimation=15 -> control at 66.7Hz
        self.decimation = 15
        self.episode_length_s = 10.0
        # Simulation settings
        self.sim.dt = 0.001
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # PhysX settings
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # Viewer settings
        self.viewer.eye = (4.0, 4.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        # Contact sensor update period
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class QminiBIRLEnvCfg_PLAY(QminiBIRLEnvCfg):
    """Play/evaluation configuration with fewer environments and no randomization."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
