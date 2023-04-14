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
    return os.path.join(
        dir,
        '{}_model_iter_{}.pth'.format(name, iteration),
    )


def eval_file_name(dir: str, name: str, iteration: int) -> str:
    return os.path.join(
        dir,
        '{}_eval_iter_{}.pth'.format(name, iteration),
    )


def dataset_file_name(dir: str, name: str, iteration: int) -> str:
    return os.path.join(
        dir,
        '{}_dataset_iter_{}.pth'.format(name, iteration),
    )


class ModelCheckpoint(typing_extensions.TypedDict):
    iteration: int
    net_state_dict: dict
    optimizer_state_dict: dict
    scheduler_state_dict: dict


def save_model(
    iteration: int,
    net_state_dict: dict,
    optimizer_state_dict: dict,
    scheduler_state_dict: dict,
    file_name: str,
) -> None:
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
    return torch.load(file_name, map_location='cpu')


class EvalCheckpoint(typing_extensions.TypedDict):
    iteration: int
    latest_avg_reward: float
    best_model_iteration: int
    best_avg_reward: float


def save_eval(
    iteration: int,
    latest_avg_reward: float,
    best_model_iteration: int,
    best_avg_reward: float,
    file_name: str,
) -> None:
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
    return torch.load(file_name)
