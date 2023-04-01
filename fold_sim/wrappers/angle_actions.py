import typing

import gymnasium as gym
from gymnasium import spaces

from fold_sim.envs.group_fold import AZIMUTHAL_DIM, POLAR_DIM


class AngleActions(gym.ActionWrapper):
    '''
    Take discretized angle values and convert to a global action index.
    '''

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.action_space = spaces.MultiDiscrete((
            AZIMUTHAL_DIM,
            POLAR_DIM,
        ))
    
    def action(self, action: typing.Sequence[int]) -> int:
        az, polar = tuple(action)
        return (az * POLAR_DIM) + polar
