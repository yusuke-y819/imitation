from gymnasium.envs.registration import register

register(
    id='RacingEnv-v0',
    entry_point='racing_gym.envs:RacingEnv',
)