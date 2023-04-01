import typing
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from fold_sim.envs.group_fold import GROUP_SIZE, RES_ENCODE_N_CAT

N_LEN_BINS = 15 # Plus one for > maximum length (30 A)
LEN_BINS = np.linspace(0, 30, N_LEN_BINS + 1)[1:]

N_DIST_BINS = 20 # Plus one for > maximum separation (40 A)
DIST_BINS = np.linspace(0, 40, N_DIST_BINS + 1)[1:]

PAST_FEATURE_SPLITS = np.add.accumulate((2, 1, 1))
FUTURE_FEATURE_SPLITS = np.add.accumulate((2, 1))

class TransformerInputObs(gym.ObservationWrapper):
    '''
    Modify the output to facilitate input to a transformer network.
    '''

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = spaces.Sequence(spaces.Box(
            -1,
            1,
            (
                # Binary past / future indicator
                1
                # Unit-scaled indices
                + 1
                + 1
                # Replace lengths with categorical buckets
                + N_LEN_BINS + 1
                # Replace distances with categorical buckets
                + N_DIST_BINS + 1
                # Unit vectors pointing towards group / aligned with
                # group vector (all zeros for future groups)
                + 1 * 3
                + 1 * 3
                # Residue sequence
                + GROUP_SIZE * RES_ENCODE_N_CAT,
            )
        ))

    def observation(
        self,
        obs: typing.Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        return np.vstack((
            np.column_stack(
                (lambda idxs, lens, distances, rest: (
                    np.zeros(len(obs[0])),
                    idxs / self.groups.size,
                    np.eye(N_LEN_BINS + 1)[np.digitize(lens.ravel(), LEN_BINS)],
                    np.eye(N_DIST_BINS + 1)[np.digitize(distances.ravel(), DIST_BINS)],
                    rest,
                ))(*np.hsplit(obs[0], PAST_FEATURE_SPLITS)),
            ),
            np.column_stack(
                (lambda idxs, lens, rest: (
                    np.ones(len(obs[1])),
                    idxs / self.groups.size,
                    np.eye(N_LEN_BINS + 1)[np.digitize(lens.ravel(), LEN_BINS)],
                    np.zeros((len(obs[1]), N_DIST_BINS + 1)),
                    np.zeros((len(obs[1]), 2 * 3)),
                    rest,
                ))(*np.hsplit(obs[1], FUTURE_FEATURE_SPLITS)),
            ),
        ))
