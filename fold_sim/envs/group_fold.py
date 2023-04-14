'''
This file defines the simplified discrete protein folding environment.

Created on 1/31/2023 by Steven Laverty (lavers@rpi.edu)
'''

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import groupby

import gymnasium as gym
import numpy as np
import quaternion
import typing_extensions
from gymnasium import spaces

from fold_sim.utils import mmcif_parsing, protein, residue_constants

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

RES_ENCODE_N_CAT = 22
GROUP_SIZE = 8
AZIMUTHAL_DIM = 24
POLAR_DIM = 12

# Use double precision for quaternion arithmetic
AZIMUTHAL_SPACE = np.linspace(
    0,
    2 * np.pi,
    AZIMUTHAL_DIM,
    endpoint=False,
    dtype=np.float64,
)
POLAR_SPACE = np.linspace(
    0,
    np.pi,
    POLAR_DIM,
    endpoint=False,
    dtype=np.float64,
)

# Get the next reference frame in terms of the previous local reference
# frame.
#
# New x axis aligned with group vector
# New y axis aligned with first side chain
# - First rotation: azimuthal rotation about previous x axis
# - Second rotation: polar rotation about the new z axis
ACTION_TO_QUAT = np.ravel(np.multiply.outer(
    quaternion.as_quat_array(np.column_stack((
        np.cos(AZIMUTHAL_SPACE / 2),
        np.sin(AZIMUTHAL_SPACE / 2),
        np.zeros(AZIMUTHAL_DIM),
        np.zeros(AZIMUTHAL_DIM),
    ))),
    quaternion.as_quat_array(np.column_stack((
        np.cos(POLAR_SPACE / 2),
        np.zeros(POLAR_DIM),
        np.zeros(POLAR_DIM),
        np.sin(POLAR_SPACE / 2),
    ))),
)
)


class Info(typing_extensions.TypedDict):
    '''
    Expose debugging info
    '''
    sim_group_locations: np.ndarray
    sim_group_rotations: np.ndarray
    true_group_locations: np.ndarray
    true_group_rotations: np.ndarray


class ResetOptions(typing_extensions.TypedDict):
    file_id: str
    chain_id: str


@dataclass(frozen=True)
class GroupSequence():
    '''
    Protein sequence is broken up into groups of residues of size
    GROUP_LEN
    '''

    size: int
    '''
    The number of residue groups in the protein sequence.
    '''

    initial_pad: int
    '''
    The number of padded residues before the first true residue in the
    first group
    '''

    final_pad: int
    '''
    The number of padded residues after the last true residue in the
    last group
    '''

    seqres: np.ndarray
    '''
    Sequence of one-hot vector blocks indicating the residue sequence
    for each residue group.

    shape=(size, GROUP_SIZE, RES_ENCODE_N_CAT), dtype=int8
    '''

    vec_len: np.ndarray
    '''
    Sequence of true (measured) lengths of each residue group's vector
    representation.

    The length is calculated as the Euclidian distance between the
    group's starting and ending coordinates.

    shape=(size), dtype=float64
    '''

    true_loc: np.ndarray
    '''
    Sequence of true (measured) coordinates of each residue group in the
    folded protein sequence.

    Index 0 is the coordinates of the first N atom in the first residue
    group. All subsequent entries are the coordinates of the last C atom
    in each residue group.
    
    shape=(size + 1, 3), dtype=float64
    '''

    true_rot: np.ndarray
    '''
    Sequence of rotations representing the true (measured) local basis
    of each residue group other than the first.

    Maps local coordinates to global coordinates.
    
    In the local basis, the x axis is aligned with the group's
    vector representation. The y axis lies in the plane formed by
    this group's vector representation and the past group's vector
    representation.

    shape=(size, 3, 3), dtype=float64
    '''


@dataclass(frozen=True)
class State():
    '''
    State variables representing the folding simulation process
    '''

    idx: int
    '''
    The index of the next group to be placed.
    '''

    sim_loc: np.ndarray
    '''
    Sequence of coordinates of each residue group that has been placed
    so far in the folding simulation.

    Index 0 is the coordinates of the first N atom in the first residue
    group. All subsequent entries are the coordinates of the last C atom
    in each residue group.
    
    shape=(size + 1, 3), dtype=float64
    '''

    sim_rot: np.ndarray
    '''
    Sequence of rotations representing the local basis of each residue
    group that has been placed so far in the folding simulation.

    Maps local coordinates to global coordinates.
    
    In the local basis, the x axis is aligned with the group's
    vector representation. The y axis lies in the plane formed by
    this group's vector representation and the past group's vector
    representation.

    shape=size, dtype=quaternion
    '''


class ResGroupFoldDiscrete(gym.Env):
    observation_space = spaces.Tuple((
        # Past observations
        spaces.Sequence(spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                # FOR EACH ALREADY-PLACED GROUP:
                # Absolute index of this group in the protein sequence.
                1
                # Relative index of this group to the current group.
                + 1
                # Length (2-norm) of the vector representation for this
                # group (angstrom).
                + 1
                # Euclidean distance from the current group's local
                # origin to this group's local origin
                + 1
                # Unit vector pointing from the current group's local
                # origin to this group's local origin, in the current
                # group's reference frame.
                + 1 * 3
                # Unit-scaled vector representation for this group
                # (angstrom)
                + 1 * 3
                # One-hot encoding of the sequence of residues in this
                # group.
                + GROUP_SIZE * RES_ENCODE_N_CAT,
            ),
            dtype=np.float64,
        )),
        # Future observations
        spaces.Sequence(spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                # Absolute index of this group in the protein sequence.
                1
                # Relative index of this group to the current group.
                + 1
                # Length (2-norm) of vector representation for this
                # group (angstrom).
                + 1
                # One-hot encoding of the sequence of residues in this
                # group.
                + GROUP_SIZE * RES_ENCODE_N_CAT,
            ),
            dtype=np.float64,
        )),
    ))

    action_space = spaces.Discrete(
        AZIMUTHAL_DIM + AZIMUTHAL_DIM * (POLAR_DIM - 1) * AZIMUTHAL_DIM
    )

    reward_range = (-np.inf, 0.0)

    seqres: str
    atom_coords: np.ndarray
    atom_mask: np.ndarray
    groups: GroupSequence
    state: State

    def __init__(self, render_mode=None) -> None:
        assert render_mode is None

        # TODO initialize caching

    def _get_obs(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        Get the current state, formatted as input for the RL agent.
        '''

        # Calculate distance kernel for residue groups that have already
        # been placed.
        displacement = self.state.sim_loc[:-1] - self.state.sim_loc[-1]
        distance = np.linalg.norm(displacement, axis=-1)
        displacement /= distance[:, None]
        vectors = (
            (self.state.sim_loc[1:] - self.state.sim_loc[:-1])
            / self.groups.vec_len[:self.state.idx][:, None]
        )
        displacement, vectors = quaternion.rotate_vectors(
            self.state.sim_rot[-1].conj(),
            (displacement, vectors),
        )
        past_obs = np.column_stack((
            np.arange(self.state.idx),
            np.arange(-self.state.idx, 0),
            self.groups.vec_len[:self.state.idx],
            distance,
            displacement,
            vectors,
            np.reshape(
                self.groups.seqres[:self.state.idx],
                (-1, GROUP_SIZE * RES_ENCODE_N_CAT)
            ),
        ))

        # List future groups that have yet to be placed.
        future_obs = np.column_stack((
            np.arange(self.state.idx, self.groups.size),
            np.arange(self.groups.size - self.state.idx),
            self.groups.vec_len[self.state.idx:],
            np.reshape(
                self.groups.seqres[self.state.idx:],
                (-1, GROUP_SIZE * RES_ENCODE_N_CAT)
            ),
        ))

        return past_obs, future_obs

    def _get_info(self) -> Info:
        '''
        Get info about the environment for debugging purposes.
        '''

        return Info(
            sim_group_locations=self.state.sim_loc,
            sim_group_rotations=self.state.sim_rot,
            true_group_locations=self.groups.true_loc,
            true_group_rotations=self.groups.true_rot,
        )

    def _get_reward(self) -> float:
        '''
        Calculate FAPE (frame-aligned point error):
        -   Using each of the true group reference frames, find the
            locations of every group in in the true reference frame and
            compare to the simulation
        -   Compute MSE for every group in every local frame, then
            return the average over all groups
        '''

        return -np.mean(np.linalg.norm(
            # Sim: group points in local frames (n_groups - 1, n_groups + 1, 3)
            (
                self.state.sim_loc[None]  # (1, n_groups + 1, 3)
                - self.state.sim_loc[1:-1, None, :]  # (n_groups - 1, 1, 3)
            )
            # Sim: rotation matrices (n_groups - 1, 3, 3)
            @ quaternion.as_rotation_matrix(self.state.sim_rot[1:])
            # True: group points in local frames (n_groups - 1, n_groups + 1, 3)
            - (
                self.groups.true_loc[None]  # (1, n_groups + 1, 3)
                - self.groups.true_loc[1:-1, None, :]  # (n_groups - 1, 1, 3)
            )
            # True: rotation matrices (n_groups - 1, 3, 3)
            @ self.groups.true_rot[1:],
            axis=-1,
        ))

        # TODO collision loss

    def reset(
        self,
        *,
        seed: int | None = None,
        options: ResetOptions | None = None,
    ) -> tuple[tuple[np.ndarray, np.ndarray], Info]:
        '''
        Reset the environment using a given pdb structure. The first
        residue group will have a randomized number of padded residues
        before the first real one.
        '''

        super().reset(seed=seed)

        logger.debug('Parsing PDB file')
        # Parse PDB mmcif file
        with open(options['file_id']) as file:
            mmcif_string = file.read()
        try:
            parse_result = mmcif_parsing.parse(
                file_id=options['file_id'],
                mmcif_string=mmcif_string,
            )
        except Exception as err:
            raise Exception('Could not parse mmcif file {}'.format(
                options['file_id'])) from err
        try:
            self.seqres = parse_result.mmcif_object.chain_to_seqres[
                options['chain_id']
            ]
            onehot_seqres = residue_constants.sequence_to_onehot(
                self.seqres,
                residue_constants.restype_order_with_x,
            )
        except Exception as err:
            raise Exception('Sequence id {} not found in file id {}'.format(
                options['chain_id'], options['file_id'])) from err
        self.atom_coords, self.atom_mask = mmcif_parsing.get_atom_coords(
            # Order: N, Cα, C, O
            parse_result.mmcif_object,
            options['chain_id'],
        )

        # Identify longest usable protein segment from atom coords mask
        # and crop
        seq_len = 0
        scan_idx = 0
        for coords_known, segment in groupby(
            self.atom_mask[:, 0].astype(np.bool8)  # N
            & self.atom_mask[:, 2].astype(np.bool8),  # C
        ):
            segment_len = sum(1 for _ in segment)
            if coords_known and segment_len > seq_len:
                seq_len = segment_len
                seq_idx = scan_idx
            scan_idx += segment_len
        self.seqres = self.seqres[seq_idx:seq_idx + seq_len]
        onehot_seqres = onehot_seqres[seq_idx:seq_idx + seq_len]
        self.atom_coords = self.atom_coords[seq_idx:seq_idx + seq_len]
        self.atom_mask = self.atom_mask[seq_idx:seq_idx + seq_len]
        logger.debug('Finished parsing PDB file')

        # Split the sequence into groups
        initial_pad = self.np_random.integers(GROUP_SIZE)
        num_groups = (
            (initial_pad + seq_len + GROUP_SIZE - 1) // GROUP_SIZE
        )
        final_pad = num_groups * GROUP_SIZE - seq_len - initial_pad
        group_seqres = np.zeros(
            (num_groups, GROUP_SIZE, RES_ENCODE_N_CAT),
            dtype=np.int8,
        )
        group_loc = np.empty((num_groups + 1, 3), dtype=np.float64)

        # First group (with padding)
        first_group_size = min(GROUP_SIZE - initial_pad, seq_len)
        group_seqres[0, :initial_pad, 0] = 1
        group_seqres[
            0,
            initial_pad:initial_pad + first_group_size,
            1:
        ] = onehot_seqres[:first_group_size]
        group_loc[0] = self.atom_coords[0, 0]  # N (First group ONLY)

        # Last group (with padding)
        last_group_size = min(GROUP_SIZE - final_pad, seq_len)
        group_seqres[
            -1,
            -(last_group_size + final_pad):GROUP_SIZE - final_pad,
            1:
        ] = onehot_seqres[-last_group_size:]
        group_seqres[-1, GROUP_SIZE - final_pad:, 0] = 1
        group_loc[-1] = self.atom_coords[-1, 2]  # C

        # Middle groups (no padding)
        if num_groups > 2:
            group_seqres[1:-1, :, 1:] = np.reshape(
                onehot_seqres[first_group_size:-last_group_size],
                (num_groups - 2, GROUP_SIZE, -1),
            )
            group_loc[1:-1] = self.atom_coords[
                first_group_size - 1:-last_group_size:GROUP_SIZE,
                2  # C
            ]

        # Calculate all group vector representation lengths
        group_vec = group_loc[1:] - group_loc[:-1]
        group_vec_len = np.linalg.norm(group_vec, axis=-1)

        # Calculate true reference frames using Gram-Schmidt process
        x_hat = group_vec / group_vec_len[:, None]
        xy_vec = x_hat[1:] - x_hat[:-1]
        # First group's reference frame is z-aligned with the second
        xy_vec = np.concatenate((
            xy_vec[0][None],
            xy_vec,
        ))
        # x_hat = x_hat[1:]
        y_vec = (
            xy_vec - x_hat * np.sum((xy_vec * x_hat), axis=-1, keepdims=True)
        )
        y_hat = y_vec / np.linalg.norm(y_vec, axis=-1, keepdims=True)
        z_hat = np.cross(x_hat, y_hat)
        group_rot = np.stack(
            (x_hat, y_hat, z_hat),
            axis=-1,
        )

        # Save group sequence
        self.groups = GroupSequence(
            size=num_groups,
            initial_pad=initial_pad,
            final_pad=final_pad,
            seqres=group_seqres,
            vec_len=group_vec_len,
            true_loc=group_loc,
            true_rot=group_rot,
        )

        # Initialize the simulation state by placing the first sequence
        # group at the global origin oriented along the positive x axis.
        self.state = State(
            idx=1,
            sim_loc=np.asarray((
                np.zeros(3, dtype=np.float64),
                np.asarray((self.groups.vec_len[0], 0, 0)),
            )),
            sim_rot=np.asarray((quaternion.one,)),
        )

        # Return the initial observation
        return self._get_obs(), self._get_info()

    def step(
        self,
        action: int,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        float,
        bool,
        bool,
        Info,
    ]:
        '''
        Place the next residue group according to a discrete action.
        Update the intermediate simulated structure and provide a
        reconstruction error reward only when the final group has been
        placed.
        '''

        # Only step if the state is not already terminal
        if self.state.idx < self.groups.size:
            new_rot = self.state.sim_rot[-1] * ACTION_TO_QUAT[action]
            new_coords = (
                self.state.sim_loc[self.state.idx]
                + quaternion.as_vector_part(
                    new_rot
                    * np.quaternion(self.groups.vec_len[self.state.idx], 0, 0)
                    * new_rot.conjugate()
                )
            )
            self.state = State(
                idx=self.state.idx + 1,
                sim_loc=np.append(
                    self.state.sim_loc,
                    (new_coords,),
                    axis=0,
                ),
                sim_rot=np.append(
                    self.state.sim_rot,
                    (new_rot,),
                    axis=0,
                ),
            )

        # Return updated observations and reward
        is_terminal = self.state.idx == self.groups.size
        return (
            self._get_obs(),
            self._get_reward() if is_terminal else 0.,
            is_terminal,
            False,
            self._get_info(),
        )

    def close(self):
        # TODO free caching
        pass

    def export_pdb(self) -> tuple[str, str]:
        '''
        Generate a tuple of pdb strings; one represents the true
        structure, the other represents the simulated structure.
        They are aligned along the reference frame of the first
        amino acid group.
        '''

        aatype = np.fromiter(
            (residue_constants.restype_order_with_x[res]
             for res in self.seqres),
            int,
        )
        residue_index = np.arange(len(aatype))
        # Export the true structure
        true_structure = protein.to_pdb(protein.from_prediction(
            {
                'aatype': aatype,
                'residue_index': residue_index,
            },
            {
                'final_atom_positions': self.atom_coords,
                'final_atom_mask': self.atom_mask,
            },
        ))
        # Export the simulated structure -- map all atoms to their simulated
        # locations.
        splits = np.add.accumulate(
            [GROUP_SIZE - self.groups.initial_pad]
            + [GROUP_SIZE] * (self.state.idx - 1)
        )
        sim_coords = self.groups.true_loc[0] + np.concatenate([
            sim_loc + quaternion.rotate_vectors(
                sim_rot,
                (atom_coords - true_loc) @ true_rot,
            )
            for sim_loc, sim_rot, true_loc, true_rot, atom_coords in zip(
                self.state.sim_loc,
                self.state.sim_rot,
                self.groups.true_loc,
                self.groups.true_rot,
                np.split(self.atom_coords, splits)[:-1]
            )
        ]) @ self.groups.true_rot[0].T
        sim_structure = protein.to_pdb(protein.from_prediction(
            {
                'aatype': aatype[:splits[-1]],
                'residue_index': residue_index[:splits[-1]],
            },
            {
                'final_atom_positions': sim_coords,
                'final_atom_mask': self.atom_mask[:splits[-1]],
            },
        ))
        return true_structure, sim_structure
