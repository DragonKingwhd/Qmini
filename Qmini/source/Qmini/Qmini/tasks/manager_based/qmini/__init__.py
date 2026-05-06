"""Qmini BIRL bipedal locomotion task registration."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Qmini-BIRL-v0",
    entry_point=f"{__name__}.qmini_env:QminiBIRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_env_cfg:QminiBIRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QminiBIRLPPORunnerCfg",
    },
)

gym.register(
    id="Qmini-BIRL-Play-v0",
    entry_point=f"{__name__}.qmini_env:QminiBIRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_env_cfg:QminiBIRLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QminiBIRLPPORunnerCfg",
    },
)
