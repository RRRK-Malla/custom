from gym.envs.registration import register

register(
    id='Drivey-v0',
    entry_point='custom_env.envs:DtestEnv',
)
