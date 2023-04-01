import typing

import gymnasium as gym
import numpy as np
import typing_extensions
from gymnasium import spaces

from fold_sim.envs.group_fold import GROUP_SIZE, RES_ENCODE_N_CAT

PAST_OBS_SPACE = spaces.Dict({
    # Absolute index of this group in the protein sequence.
    'abs_idx': spaces.Box(low=0, high=np.inf, dtype=np.int64),
    # Relative index of this group to the current group.
    'rel_idx': spaces.Box(low=-np.inf, high=-1, dtype=np.int64),
    # Length (2-norm) of vector representation for this group
    # (angstrom).
    'vec_len': spaces.Box(low=0, high=np.inf, dtype=np.float64),
    # A pair of Euclidean distances from the current group's local
    # origin to this group's vector representation start and end points.
    'rel_dist': spaces.Box(
        low=0, high=np.inf, shape=(2,), dtype=np.float64
    ),
    # Unit vectors pointing from the current group's local origin to this
    # group's vector representation start and end points.
    'rel_dir': spaces.Box(
        low=-1, high=1, shape=(2, 3), dtype=np.float64,
    ),
    # One-hot encoding of the sequence of residues in this group.
    'seqres': spaces.MultiBinary((GROUP_SIZE, RES_ENCODE_N_CAT)),
})

FUTURE_OBS_SPACE = spaces.Dict({
    # Absolute index of this group in the protein sequence.
    'abs_idx': spaces.Box(low=0, high=np.inf, dtype=np.int64),
    # Relative index of this group to the current group.
    'rel_idx': spaces.Box(low=0, high=np.inf, dtype=np.int64),
    # Length (2-norm) of vector representation for this group
    # (angstrom).
    'vec_len': spaces.Box(low=0, high=np.inf, dtype=np.float64),
    # One-hot encoding of the sequence of residues in this group.
    'seqres': spaces.MultiBinary((GROUP_SIZE, RES_ENCODE_N_CAT)),
})


class PastObs(typing_extensions.TypedDict):
    abs_idx: np.ndarray
    rel_idx: np.ndarray
    vec_len: np.ndarray
    rel_dist: np.ndarray
    rel_dir: np.ndarray
    seqres: np.ndarray


class FutureObs(typing_extensions.TypedDict):
    abs_idx: np.ndarray
    rel_idx: np.ndarray
    vec_len: np.ndarray
    seqres: np.ndarray


class Obs(typing_extensions.TypedDict):
    past: typing.List[PastObs]
    future: typing.List[FutureObs]


class ObservationDict(gym.ObservationWrapper):
    '''
    Reformat the observation space into a human-readable format.
    '''

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = spaces.Dict({
            'past': PAST_OBS_SPACE,
            'future': FUTURE_OBS_SPACE,
        })

    def observation(
        self,
        obs: typing.Tuple[np.ndarray, np.ndarray],
    ) -> Obs:
        return Obs(
            past=[
                PastObs(
                    abs_idx=abs_idx.astype(np.int64),
                    rel_idx=rel_idx.astype(np.int64),
                    vec_len=vec_len,
                    rel_dist=rel_dist,
                    rel_dir=rel_dir.reshape((2, 3)),
                    seqres=seqres.astype(np.bool8).reshape((
                        GROUP_SIZE, RES_ENCODE_N_CAT
                    )),
                ) for (
                    abs_idx,
                    rel_idx,
                    vec_len,
                    rel_dist,
                    rel_dir,
                    seqres,
                ) in zip(*np.hsplit(
                    obs[0],
                    np.add.accumulate((1, 1, 1, 2, 2 * 3)),
                ))
            ],
            future=[
                FutureObs(
                    abs_idx=abs_idx.astype(np.int64),
                    rel_idx=rel_idx.astype(np.int64),
                    vec_len=vec_len,
                    seqres=seqres.astype(np.bool8).reshape((
                        GROUP_SIZE, RES_ENCODE_N_CAT
                    )),
                ) for (
                    abs_idx,
                    rel_idx,
                    vec_len,
                    seqres,
                ) in zip(*np.hsplit(
                    obs[1],
                    np.add.accumulate((1, 1, 1)),
                ))
            ],
        )
