from gymnasium.envs.registration import register

register(
    id='fold_sim/ResGroupFoldDiscrete-v0',
    entry_point='fold_sim.envs:ResGroupFoldDiscrete',
)
