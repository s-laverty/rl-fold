'''
This file contains various utility functions which are used in other
files. 

Created on 3/9/2023 by Steven Laverty (lavers@rpi.edu)
'''

from __future__ import annotations
import os

import typing

import gymnasium as gym
import numpy as np
import torch
import typing_extensions


class SequenceId(typing_extensions.TypedDict):
    '''
    Unique identifying information for a protein sequence in the pdb.
    '''
    file_id: str
    chain_id: str


class Config(typing_extensions.TypedDict):
    '''
    JSON config file data.
    '''

    # Basic config
    name: str
    ''' [Required] The configuration's unique name. '''
    method: str
    ''' [Required] The RL algorithm (either alpha-zero or deep-q). '''
    pbd_path: str
    '''
    [Required] The absolute or relative path to the pbd data directory.
    '''
    sequences: list[SequenceId]
    ''' [Required] The sequence dataset to train on. '''
    num_sims_train: int
    '''
    [Required] The number of simulations to run for each sequence in the
    training dataset during model training.
    '''
    num_sims_eval: int
    '''
    [Required] The number of simulations to run for each sequence in the
    training dataset during model evaluation.
    '''
    batch_size: int
    ''' [Required] The batch size to use during training. '''
    num_batches: int
    ''' [Required] How many batches to train on before checkpointing. '''
    simulate_shm: bool
    '''
    [Optional] If true, use shared memory to transfer simulation worker
    data to the main process. Default true
    '''
    dataset_shm: bool
    '''
    [Optional] If true, keep all data in shared memory. Requires a large
    number of file descriptors. Default false.
    '''

    # Optim config
    lr: float
    ''' [Required] The learning rate. '''
    gamma: float
    ''' [Required] The learning rate scheduler decay rate. '''
    milestones: list[int]
    '''
    [Required] The model iterations at which to decay the learning rate.
    '''
    betas: tuple[int, int]
    '''
    [Optional] The betas for the ADAM W optimizer. Default [0.9, 0.999].
    '''
    l2_reg: float
    '''
    [Optional] The weight decay parameter for all non-bias model
    parameters. Default 0.
    '''
    amsgrad: bool
    '''
    [Optional] If true, use the amsgrad variant of ADAM W.
    Default false.
    '''

    # Deep Q learning config (only used when method='deep-q')
    q_epsilon_max: float
    '''
    [Required] The maximum (initial) epsilon for the epsilon-greedy
    policy.
    '''
    q_epsilon_min: float
    '''
    [Required] The minimum (final) epsilon for the epsilon-greedy
    policy.
    '''
    q_epsilon_gamma: float
    ''' [Required] The epsilon decay rate. '''
    q_learning_rate: float
    '''
    [Required] The Q-learning learning rate (alpha parameter) used for
    determining the training target Q-value.
    '''
    q_discount_factor: float
    '''
    [Required] The Q-learning discount factor (gamma parameter) used for
    determining the training target Q-value.
    '''
    q_tau: float
    '''
    [Required] The soft-update parameter for the target network.
    '''
    q_reward_backfill: float
    '''
    [Optional] The reward backfill parameter to use for each training
    episode. The reward is propagated as:
    
    (final_reward) * reward_backfill^(1 - step/num_epsiode_steps)
    '''


class TensorObs(gym.ObservationWrapper):
    '''
    Map the output to a shared torch float tensor.
    '''

    def observation(
        self,
        obs: np.ndarray,
    ) -> torch.Tensor:
        return torch.from_numpy(obs).float()


def pad_sequence_with_mask(
    seq: typing.Iterable[torch.Tensor],
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Combine tensors with different lengths into a batch.
    Also return a mask indicating which values are padded.

    Modified from torch.nn.utils.rnn.pad_sequence
    '''

    max_len = max([s.size(0) for s in seq])
    trailing_dims = seq[0].size()[1:]
    out_dims = (max_len, len(seq)) + trailing_dims
    mask_dims = (len(seq), max_len)
    out_tensor = seq[0].new_zeros(out_dims)
    mask = seq[0].new_zeros(mask_dims, dtype=torch.bool)
    for i, tensor in enumerate(seq):
        length = tensor.size(0)
        out_tensor[:length, i] = tensor
        mask[i, length:].fill_(True)
    if device is not None:
        out_tensor = out_tensor.to(device)
        mask = mask.to(device)
    return out_tensor, mask


def model_file_name(dir: str, name: str, iteration: int) -> str:
    '''
    Get the full formatted model file path for a given iteration.
    '''
    return os.path.join(
        dir,
        '{}_model_iter_{}.pth'.format(name, iteration),
    )


def eval_file_name(dir: str, name: str, iteration: int) -> str:
    '''
    Get the full formatted model eval file path for a given iteration.
    '''
    return os.path.join(
        dir,
        '{}_eval_iter_{}.pth'.format(name, iteration),
    )


def dataset_file_name(dir: str, name: str, iteration: int) -> str:
    '''
    Get the full formatted model data path for a given iteration.
    '''
    return os.path.join(
        dir,
        '{}_dataset_iter_{}.pth'.format(name, iteration),
    )


class ModelCheckpoint(typing_extensions.TypedDict):
    iteration: int
    ''' Latest model iteration. '''
    net_state_dict: dict | tuple[dict, dict]
    '''
    zero_net state dict or tuple of [policy, target] q_net state dicts
    '''
    optimizer_state_dict: dict
    ''' Optimizer state dict. '''
    scheduler_state_dict: dict
    ''' LR scheduler state dict. '''


def save_model(
    iteration: int,
    net_state_dict: dict | tuple[dict, dict],
    optimizer_state_dict: dict,
    scheduler_state_dict: dict,
    file_name: str,
) -> None:
    '''
    Save a model checkpoint given the current state.
    '''
    dirname = os.path.dirname(file_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(
        ModelCheckpoint(
            iteration=iteration,
            net_state_dict=net_state_dict,
            optimizer_state_dict=optimizer_state_dict,
            scheduler_state_dict=scheduler_state_dict,
        ),
        file_name,
    )


def load_model(file_name: str) -> ModelCheckpoint:
    '''
    Load a model checkpoint, mapping all tensors to the cpu.
    '''
    return torch.load(file_name, map_location='cpu')


class EvalCheckpoint(typing_extensions.TypedDict):
    iteration: int
    ''' Latest model iteration. '''
    latest_avg_reward: float
    ''' Latest evaluation. '''
    best_model_iteration: int
    ''' Model iteration with the best evaluation. '''
    best_avg_reward: float
    ''' Evaluation of the best model. '''


def save_eval(
    iteration: int,
    latest_avg_reward: float,
    best_model_iteration: int,
    best_avg_reward: float,
    file_name: str,
) -> None:
    '''
    Save an evaluation checkpoint given the current state.
    '''
    dirname = os.path.dirname(file_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(
        EvalCheckpoint(
            iteration=iteration,
            latest_avg_reward=latest_avg_reward,
            best_model_iteration=best_model_iteration,
            best_avg_reward=best_avg_reward,
        ),
        file_name,
    )


def load_eval(file_name: str) -> EvalCheckpoint:
    '''
    Load an evaluation checkpoint.
    '''
    return torch.load(file_name)
