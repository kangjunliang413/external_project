"""Register the external Go2 flat-terrain training and playback tasks."""

from locolab.utils.gym_registration import register_manager_based_rl_env

_GO2_CONFIG = "locolab.tasks.manager_based.locomotion.velocity.config.unitree_go2"

register_manager_based_rl_env(
    task_id="My-Velocity-Flat-Go2",
    env_cfg_module=f"{_GO2_CONFIG}.flat_env_cfg",
    env_cfg_name="Go2FlatEnvCfg",
    agent_cfg_entry_points={
        "rsl_rl_cfg_entry_point": f"{_GO2_CONFIG}.agents.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
        "z_rl_cfg_entry_point": f"{_GO2_CONFIG}.agents.z_rl_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)
