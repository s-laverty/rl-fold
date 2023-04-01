import gymnasium as gym

from fold_sim.envs.group_fold import GROUP_SIZE, RES_ENCODE_N_CAT

SCALE_COEFF = 1.2e0

class NormalizedRewards(gym.RewardWrapper):
    '''
    Normalize the rewards by the length of the protein sequence.

    Attempt to scale expected rewards to O(1)

    Expected value of distance for a random walk is O(sqrt N).
    Therefore, the expected distance^2 increases to the order of O(N).
    '''

    def reward(
        self,
        reward: float,
    ) -> float:
        return SCALE_COEFF * reward / self.groups.size
