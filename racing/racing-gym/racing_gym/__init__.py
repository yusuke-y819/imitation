from gymnasium.envs.registration import register

register(
    id='RacingEnv-v0',
    entry_point='racing_gym.envs:RacingEnv',
)

register(
    id='RacingEnv-v1',
    entry_point='racing_gym.envs:RacingEnv1',
)

register(
    id='RacingEnv-v2',
    entry_point='racing_gym.envs:RacingEnv2',
)

register(
    id='RacingEnv-v3',
    entry_point='racing_gym.envs:RacingEnv3',
)

register(
    id='RacingEnv-v4',
    entry_point='racing_gym.envs:RacingEnv4',
)