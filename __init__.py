from gymnasium.envs.registration import register

register(
    id="gymnasium_env/main",
    entry_point="gymnasium_env.envs:Env1x2",
    max_episode_steps=2000,
)
