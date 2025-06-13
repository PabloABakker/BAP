# BAP/Custom/__init__.py

from gymnasium.envs.registration import register

register(
    id="CustomDynamicsEnv-v3",
    entry_point="Custom.CustomDynamicsEnv_v3:CustomDynamicsEnv",
)
