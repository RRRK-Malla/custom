register(
    id='Drivey-v1',
    entry_point='custom_envs.envs:DtestEnv',
    reward_threshold=8, # optimum = 8.46
    max_episode_steps=200,
)
