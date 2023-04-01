#!/usr/bin/env python

'''
This file defines the procedures for training and evaluating models.

Created on 3/4/2023 by Steven Laverty (lavers@rpi.edu)
'''

from __future__ import annotations

import argparse
from collections import deque
import itertools
import json
import logging
import os
import pathlib
import random
import re
import tempfile
import time
import typing

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.optim as optim
import torch.utils.data as data
from torch.nn.parallel import DistributedDataParallel as DDP

import utils
from model.fold_q_net import FoldQNet
from model.fold_zero_net import FoldZeroNet
from simulate import batch_sim

logging.basicConfig(
    format='%(asctime)s %(filename)s [%(levelname)s]: %(message)s',
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

NUM_PREV_DATASETS = 20


def create_initial_model(
    config: dict,
    model_dir: str,
    clobber: bool,
) -> None:
    # Generate model checkpoint using config parameters.
    model_file = utils.model_file_name(
        model_dir,
        config['name'],
        0,
    )
    if not clobber and os.path.exists(model_file):
        raise ValueError('Model checkpoint for iteration 0 already exists.')
    logger.info('Creating initial model.')
    if config['method'] == 'alpha-zero':
        net = FoldZeroNet()
        state_dict = net.state_dict()
    elif config['method'] == 'deep-q':
        net = FoldQNet()
        state_dict = (
            net.state_dict(),
            dict(
                (key, weights.detach().clone())
                for key, weights in net.state_dict().items()
            )
        )
    optimizer = optim.AdamW([
        {
            'params': params,
            'lr': config['lr'],
            'betas': config.get('betas', [0.9, 0.999]),
            # Don't use weight decay on bias terms
            'weight_decay': config.get('l2_reg', 0) if 'bias' not in name else 0,
            'amsgrad': config.get('amsgrad', False),
        }
        for name, params in net.named_parameters()
    ])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['milestones'],
        gamma=config['gamma'],
    )
    utils.save_model(
        0,
        state_dict,
        optimizer.state_dict(),
        scheduler.state_dict(),
        model_file,
    )
    logger.info('Initial model checkpoint saved.')


def alpha_zero_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs, p, v = zip(*batch)
    obs, mask = utils.pad_sequence_with_mask(obs)
    p = torch.stack(p)
    v = torch.as_tensor(v)
    return obs, mask, p, v


def deep_q_collate_fn(
    batch: list[tuple[torch.Tensor, int, float, torch.Tensor, bool]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs, a, reward, next_obs, done = zip(*batch)
    obs, mask = utils.pad_sequence_with_mask(obs)
    next_obs = rnn.pad_sequence(next_obs)
    a = torch.as_tensor(a)
    reward = torch.as_tensor(reward)
    done = torch.as_tensor(done)
    return obs, a, reward, next_obs, done, mask


def alpha_zero_loss_fn(
    log_p_pred: torch.Tensor,
    v_pred: torch.Tensor,
    p: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    v_err = (v_pred - v)**2
    p_err = -torch.sum(p * log_p_pred, dim=1)
    return torch.mean(v_err + p_err)


deep_q_loss_fn = nn.SmoothL1Loss()


def train_worker(
    rank: int,
    world_size: int,
    init_file: str,
    config: dict,
    model_dir: str,
    iteration: int,
    dataset_src: str | typing.Sequence,
):
    dist.init_process_group(
        'nccl',
        init_method='file://' + init_file,
        rank=rank,
        world_size=world_size,
    )

    # Initialize distributed model with previous checkpoint
    checkpoint = utils.load_model(utils.model_file_name(
        model_dir,
        config['name'],
        iteration - 1,
    ))
    device = torch.device('cuda', rank)
    if config['method'] == 'alpha-zero':
        net = FoldZeroNet().to(device)
        state_dict = checkpoint['net_state_dict']
        collate_fn = alpha_zero_collate_fn
    elif config['method'] == 'deep-q':
        net = FoldQNet().to(device)
        target_net = FoldQNet()
        target_net.load_state_dict(checkpoint['net_state_dict'][1])
        target_net.to(device).eval()
        state_dict = checkpoint['net_state_dict'][0]
        collate_fn = deep_q_collate_fn
    net = DDP(net, device_ids=(rank,))
    net.module.load_state_dict(state_dict)
    net.to(device)
    net.train()

    # Initialize optimizer
    optimizer = optim.AdamW([{'params': params} for params in net.parameters()])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Initialize scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, ())
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Initialize dataloader
    dataset = (
        dataset_src
        if config.get('dataset_shm', False) else
        torch.load(dataset_src)
    )
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'] // world_size,
        sampler=data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        ),
        collate_fn=collate_fn,
    )

    # Train net
    for _ in range(config['num_batches']):
        optimizer.zero_grad()
        batch = next(iter(dataloader))
        if config['method'] == 'alpha-zero':
            obs, mask, p, v = map(lambda x: x.to(device), batch)
            log_p_pred, v_pred = net(obs, mask)
            loss = alpha_zero_loss_fn(log_p_pred, v_pred, p, v)
        elif config['method'] == 'deep-q':
            obs, a, reward, next_obs, done, mask = batch
            obs, reward, next_obs, done, mask = map(
                lambda x: x.to(device),
                (obs, reward, next_obs, done, mask),
            )
            pred_q = net(obs, mask)[range(mask.size(0)), a]
            with torch.no_grad():
                next_obs_q = target_net(next_obs, mask)
            val = (
                reward
                + torch.logical_not(done)
                    * config['q_discount_factor']
                    * torch.max(next_obs_q, dim=1)[0]
            )
            target_q = (
                (1 - config['q_learning_rate']) * pred_q
                + config['q_learning_rate'] * val
            )
            loss = deep_q_loss_fn(pred_q, target_q)
        if torch.isnan(loss).item():
            logger.warning('Loss is Nan!')
        loss.backward()
        optimizer.step()
        if config['method'] == 'deep-q':
            # Soft update target network
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = net.module.state_dict()
            for key in target_net_state_dict:
                target_net_state_dict[key] = (
                    config['q_tau'] * policy_net_state_dict[key]
                    + (1 - config['q_tau']) * target_net_state_dict[key]
                )
    scheduler.step()

    # Save checkpoint (rank 0 only)
    if rank == 0:
        if config['method'] == 'alpha-zero':
            state_dict = net.module.state_dict()
        elif config['method'] == 'deep-q':
            state_dict = (net.module.state_dict(), target_net.state_dict())
        utils.save_model(
            iteration,
            state_dict,
            optimizer.state_dict(),
            scheduler.state_dict(),
            utils.model_file_name(
                model_dir,
                config['name'],
                iteration,
            ),
        )

    dist.destroy_process_group()


def shuffle_combined_dataset(
    combined_dataset: typing.Iterable[typing.Iterable],
    seed: int = None,
) -> list:
    dataset = list(itertools.chain.from_iterable(combined_dataset))
    logger.debug('Combined dataset contains %d entries.', len(dataset))

    # Randomly shuffle the processed dataset
    if seed is not None:
        random.seed(seed)
    random.shuffle(dataset)
    return dataset


def train_iteration(
    config: dict,
    model_dir: str,
    iteration: int,
    combined_dataset: typing.Iterable[typing.Iterable],
    clobber: bool,
):
    # Make sure devices are available for training.
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required for net training!')
    if not dist.is_available() and dist.is_nccl_available():
        raise RuntimeError('NCCL required for net training!')
    world_size = torch.cuda.device_count()

    if not clobber and os.path.exists(utils.model_file_name(
        model_dir,
        config['name'],
        iteration,
    )):
        raise ValueError(
            'Model checkpoint for iteration {} already exists.'.format(iteration))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Collect dataset
        logger.info('Shuffling combined dataset')
        if config.get('dataset_shm', False):
            logger.debug('Processing shared dataset')
            dataset = shuffle_combined_dataset(combined_dataset)
        else:
            dataset = os.path.join(temp_dir, 'dataset.pth')
            torch.save(
                shuffle_combined_dataset(combined_dataset),
                dataset,
            )
        # Spawn training processes
        init_file = os.path.join(temp_dir, 'init')
        logger.info('Training model iteration %d', iteration)
        mp.spawn(
            train_worker,
            args=(
                world_size,
                init_file,
                config,
                model_dir,
                iteration,
                dataset,
            ),
            nprocs=world_size,
            join=True,
        )
    logger.info('Model checkpoint for iteration %d saved.', iteration)


def eval_iteration(
    config: dict,
    model_dir: str,
    evaluation_dir: str,
    iteration: int,
    prev_best_iteration: int,
    prev_best_avg_reward: float,
    clobber: bool,
) -> tuple[int, float]:
    eval_file = utils.eval_file_name(
        evaluation_dir,
        config['name'],
        iteration,
    )
    if not clobber and os.path.exists(eval_file):
        raise ValueError(
            'Evaluation checkpoint for iteration {} already exists.'.format(iteration))
    logger.info('Evaluating model iteration %d', iteration)
    results = batch_sim(
        config,
        iteration,
        utils.model_file_name(
            model_dir,
            config['name'],
            iteration,
        ),
        evaluate=True,
    )
    if config['method'] in ('alpha-zero', 'deep-q'):
        avg_reward = sum(val for _, _, val in results) / len(results)
    if avg_reward > prev_best_avg_reward:
        best_iteration = iteration
        best_avg_reward = avg_reward
        logger.info(
            'Model iteration %d (average reward %.4f) outperforms all previous model iterations.', iteration, avg_reward)
    else:
        best_iteration = prev_best_iteration
        best_avg_reward = prev_best_avg_reward
        logger.info('Model iteration %d (average reward %.4f) did not outperform the previous best model (iteration %d, average reward %.4f).',
                    iteration, avg_reward, prev_best_iteration, prev_best_avg_reward)
    utils.save_eval(
        iteration,
        avg_reward,
        best_iteration,
        best_avg_reward,
        eval_file,
    )
    logger.info('Evaluation checkpoint for iteration %d saved.', iteration)

    # Return the best iteration
    return best_iteration, best_avg_reward


def process_raw_dataset(
    config: dict,
    raw_dataset: typing.Iterable,
) -> list:
    if config['method'] == 'alpha-zero':
        return [
            (obs, p, value)
            for observations, policies, value in raw_dataset
            for obs, p in zip(observations, policies)
        ]
    if config['method'] == 'deep-q':
        backfill = config.get('q_reward_backfill', 0)
        return [
            (obs_pair[..., 0], action, reward, obs_pair[..., 1], is_done)
            for observations, actions, value in raw_dataset
            for obs_pair, action, reward, is_done in zip(
                observations.unfold(0, 2, 1),
                actions.long(),
                value * (backfill**torch.linspace(1, 0, len(actions))),
                (False,) * (len(actions) - 1) + (True,),
            )
        ]


def gen_dataset(
    config: dict,
    model_file: str,
    dataset_dir: str,
    iteration: int,
    clobber: bool,
) -> list:
    dataset_file = utils.dataset_file_name(
        dataset_dir,
        config['name'],
        iteration,
    )
    if not clobber and os.path.exists(dataset_file):
        raise ValueError(
            'Dataset for iteration {} already exists.'.format(iteration))
    logger.info(
        'Generating dataset for iteration %d.', iteration)
    dataset = batch_sim(
        config,
        iteration,
        model_file,
        dataset_file,
    )
    logger.info('Dataset for iteration %d saved.', iteration)
    return process_raw_dataset(
        config,
        dataset,
    )


def collect_datasets(
    config: dict,
    dataset_dir: str,
    iteration: int,
    init: bool = False,
) -> list[list]:
    # Load datasets from the most recent model iterations into shared memory
    combined_dataset = []
    # last_iteration = iteration - 1 if init else iteration
    for i in range(max(iteration - NUM_PREV_DATASETS, 0), iteration):
        dataset_file = utils.dataset_file_name(dataset_dir, config['name'], i)
        if os.path.exists(dataset_file):
            dataset = torch.load(dataset_file)
            if config.get('dataset_shm', False):
                dataset[1].share_memory_()
                combined_dataset.append(process_raw_dataset(
                    config,
                    dataset[0],
                ))
            else:
                combined_dataset.append(process_raw_dataset(
                    config,
                    dataset,
                ))
        elif not init or i < iteration - 1:
            logger.warning('Missing dataset for iteration %d.', i)
    return combined_dataset


def train_pipeline(
    config: dict,
    start_iteration: int,
    num_iterations: int,
    model_dir: str,
    evaluation_dir: str,
    dataset_dir: str,
    clobber: bool,
    purge: bool,
) -> None:

    if start_iteration == 0:
        # Create initial model if start_iteration is 0
        create_initial_model(config, model_dir, clobber)
        best_iteration, best_avg_reward = eval_iteration(
            config,
            model_dir,
            evaluation_dir,
            0,
            -1,
            -float('inf'),
            clobber,
        )
        # There shouldn't be a dataset for the new model yet.
        dataset_file = utils.dataset_file_name(dataset_dir, config['name'], 0)
        if os.path.exists(dataset_file):
            if not clobber:
                raise ValueError('Dataset for iteration 0 already exists.')
            os.remove(dataset_file)
        start_iteration += 1
    else:
        # Make sure the previous model exists.
        if not os.path.exists(utils.model_file_name(
            model_dir,
            config['name'],
            start_iteration - 1,
        )):
            raise ValueError('Cannot resume training at iteration {}. Missing model checkpoint for iteration {}.'.format(
                start_iteration, start_iteration - 1))

        # Make sure the previous model was evaluated. If not, evaluate it now.
        eval_file = utils.eval_file_name(
            evaluation_dir,
            config['name'],
            start_iteration - 1,
        )
        if not os.path.exists(eval_file):
            logger.info(
                'Evaluation checkpoint for iteration %d not found. Evaluating it now.', start_iteration - 1)
            prev_eval_file = utils.eval_file_name(
                evaluation_dir,
                config['name'],
                start_iteration - 2,
            )
            if not os.path.exists(prev_eval_file):
                raise ValueError('Cannot evaluate model iteration {}. Missing evaluation checkpoint for iteration {}.'.format(
                    start_iteration - 1, start_iteration - 2))
            prev_checkpoint = utils.load_eval(prev_eval_file)
            best_iteration, best_avg_reward = eval_iteration(
                config,
                model_dir,
                evaluation_dir,
                start_iteration - 1,
                prev_checkpoint['best_model_iteration'],
                prev_checkpoint['best_avg_reward'],
                clobber,
            )
            del prev_checkpoint
        else:
            checkpoint = utils.load_eval(eval_file)
            best_iteration = checkpoint['best_model_iteration']
            best_avg_reward = checkpoint['best_avg_reward']
            del checkpoint
    
    # Load most recent datasets
    combined_dataset = deque(
        collect_datasets(
            config,
            dataset_dir,
            start_iteration,
            True,
        ),
        maxlen=NUM_PREV_DATASETS,
    )

    # Begin main training cycle
    for iteration in range(start_iteration, start_iteration + num_iterations):
        # Generate training data
        # --
        # For the start iteration only, skip this step if the dataset
        # already exists (can happen if the previous training cycle was
        # interrupted).
        if iteration != start_iteration or not os.path.exists(
            utils.dataset_file_name(
                dataset_dir,
                config['name'],
                iteration - 1,
            ),
        ):
            # For alpha-zero, use the best model
            if config['method'] == 'alpha-zero':
                model_file = utils.model_file_name(
                    model_dir,
                    config['name'],
                    best_iteration,
                )
            # For deep q, use the most recent model
            elif config['method'] == 'deep-q':
                model_file = utils.model_file_name(
                    model_dir,
                    config['name'],
                    iteration - 1,
                )
            combined_dataset.append(gen_dataset(
                config,
                model_file,
                dataset_dir,
                iteration - 1,
                clobber,
            ))
            if purge and iteration > NUM_PREV_DATASETS:
                old_dataset = utils.dataset_file_name(
                    dataset_dir,
                    config['name'],
                    iteration - NUM_PREV_DATASETS - 1,
                )
                if os.path.exists(old_dataset):
                    os.remove(old_dataset)
                logger.info('Dataset iteration %d removed.', iteration - NUM_PREV_DATASETS - 1)


        # Train this iteration.
        train_iteration(
            config,
            model_dir,
            iteration,
            combined_dataset,
            clobber,
        )

        # Evaluate the new model.
        best_iteration, best_avg_reward = eval_iteration(
            config,
            model_dir,
            evaluation_dir,
            iteration,
            best_iteration,
            best_avg_reward,
            clobber,
        )


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str,
                        help='Configuration JSON file used for training.')
    parser.add_argument('num_iterations', nargs='?', default=1, type=int,
                        help='Number of training iterations (%(default)s by default).')
    parser.add_argument('-i', '--iteration', default=0, type=int,
                        help='Current iteration number to resume training (%(default)s by default).', metavar='N')
    parser.add_argument('--data-dir', default='datasets',
                        help='Directory where datasets are stored ("%(default)s" by default).', metavar='DIRNAME')
    parser.add_argument('--model-dir', default='models',
                        help='Directory where model parameters are stored ("%(default)s" by default).', metavar='DIRNAME')
    parser.add_argument('--eval-dir', default='model_evaluations',
                        help='Directory where model evaluation checkpoints are stored ("%(default)s" by default).', metavar='DIRNAME')
    parser.add_argument('-c', '--clobber', action='store_true',
                        help='Overwrite existing model and dataset files.')
    parser.add_argument('-a', '--auto', action='store_true',
                        help='Automatically continue from the next previous model iteration found in the model directory.')
    parser.add_argument('-p', '--purge', action='store_true', help='Automatically purge old (unused) dataset files as they become unneeded.')
    args = parser.parse_args()

    # Use the provided configuration file to determine training hyperparameters
    logger.info('Parsing config file.')
    with open(args.config_file) as f:
        config = json.load(f)
    if args.auto:
        if args.iteration != 0:
            raise ValueError(
                'Using the -a flag is mutually exclusive with using the -i flag!')

        for fname in pathlib.Path(args.model_dir).glob(
            utils.model_file_name('.', config['name'], '*')
        ):
            iteration = int(re.findall(r'\d+', str(fname))[-1])
            args.iteration = max(args.iteration, iteration + 1)

    # Start train pipeline
    logger.info('Starting training pipeline at iteration %d.', args.iteration)
    train_pipeline(
        config=config,
        start_iteration=args.iteration,
        num_iterations=args.num_iterations,
        model_dir=args.model_dir,
        evaluation_dir=args.eval_dir,
        dataset_dir=args.data_dir,
        clobber=args.clobber,
        purge=args.purge,
    )
    logger.info('Training pipeline complete.')
