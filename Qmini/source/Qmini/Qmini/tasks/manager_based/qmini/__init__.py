"""Qmini walking task registration."""

import gymnasium as gym

from . import agents


gym.register(
    id="Qmini-Walk-v0",
    entry_point=f"{__name__}.qmini_env:QminiWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_env_cfg:QminiWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QminiWalkPPORunnerCfg",
    },
)

gym.register(
    id="Qmini-Walk-Play-v0",
    entry_point=f"{__name__}.qmini_env:QminiWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_env_cfg:QminiWalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QminiWalkPPORunnerCfg",
    },
)
