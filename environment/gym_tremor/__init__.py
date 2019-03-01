from gym.envs.registration import register

register(
    id='tremor-v0',
    entry_point='environment.gym_tremor.envs:TremorEnv',
)