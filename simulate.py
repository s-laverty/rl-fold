'''
This file defines the procedure for generating simulation data to be used
for training and evaluation.

Created on 3/3/2023 by Steven Laverty (lavers@rpi.edu)
'''

from __future__ import annotations
from contextlib import nullcontext
import ctypes
import json
import logging
import os
import queue
import tempfile
import typing
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Barrier

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from model.fold_q_net import FoldQNet

import fold_sim
from utils import TensorObs, load_model, pad_sequence_with_mask
from fold_sim.envs.group_fold import AZIMUTHAL_DIM, POLAR_DIM, State
from fold_sim.wrappers import NormalizedRewards, TransformerInputObs
from model.fold_zero_net import FoldZeroNet
from MCTS import UCT_search

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

POLAR_ONLY = np.arange(POLAR_DIM)
ALL_ACTIONS = np.arange(AZIMUTHAL_DIM * POLAR_DIM)
MAX_OBS_LEN = 128
MAX_CONCURRENT_EVALS = 64


def run_alphazero_sim(
    seq_id: dict[str, str],
    net: typing.Callable[[torch.Tensor], tuple[np.ndarray, float]],
    evaluate: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor], float]:
    def get_actions(state: State):
        if state.idx == 1:
            return POLAR_ONLY
        return ALL_ACTIONS

    # Initialize environment
    env = gym.make('fold_sim/ResGroupFoldDiscrete-v0')
    env = TransformerInputObs(env)
    env = TensorObs(env)
    env = NormalizedRewards(env)

    # Run simulation
    observations = []
    policies = []
    obs, _ = env.reset(options=seq_id)
    is_done = False
    root = None
    while not is_done:
        # Do a MCTS from the current state
        logger.debug('Start UCT search (%s)', mp.current_process().name)
        root = UCT_search(
            env,
            net,
            get_actions,
            obs,
            800,
            root,
            bias=-1.0,
        )
        logger.debug('End UCT search (%s)', mp.current_process().name)

        # Get the policy.
        # For the first 40% of moves, use a temperature of 1;
        # then use infinitesimal temperature eps
        frac = env.unwrapped.state.idx / env.unwrapped.groups.size
        temp = 1.0 if not evaluate and frac < 0.40 else 1e-1
        p_mask = root.child_number_visits**(1 / temp)
        p_mask /= np.sum(p_mask)
        p = np.zeros_like(ALL_ACTIONS, dtype=np.float64)
        p[root.actions] = p_mask
        observations.append(obs)
        policies.append(torch.from_numpy(p).float())

        # Take an action according to the policy
        a = np.random.choice(root.actions, p=p_mask)
        obs, value, is_done, _, _ = env.step(a)
        root = root.children[a]

    # Return this simulation's data
    return observations, policies, value


def run_q_sim(
    seq_id: dict[str, str],
    net: typing.Callable[[torch.Tensor], torch.Tensor],
    evaluate: bool = False,
    epsilon: float = 0.01,
) -> tuple[list[torch.Tensor], list[int], float]:
    def get_actions(obs: torch.Tensor):
        # For the first action only, restrict to polar transformations
        # If the second group has already been placed, then its future
        # indicator will be zero.
        if obs[1, 0]:
            return POLAR_ONLY
        return ALL_ACTIONS

    # Initialize environment
    env = gym.make('fold_sim/ResGroupFoldDiscrete-v0')
    env = TransformerInputObs(env)
    env = TensorObs(env)
    env = NormalizedRewards(env)

    # Run simulation
    observations = []
    actions = []
    obs, _ = env.reset(options=seq_id)
    done = False
    while not done:
        # Use epsilon-greedy policy
        if not evaluate and np.random.rand() < epsilon:
            a = np.random.choice(get_actions(obs))
        else:
            q_values = net(obs)
            a = torch.argmax(q_values[get_actions(obs)]).item()
        observations.append(obs)
        actions.append(a)
        obs, value, done, _, _ = env.step(a)
    # Append terminal state observation and return this simulation's data
    observations.append(obs)
    return observations, actions, value


def net_worker(
    method: str,
    model_file: str,
    device: torch.device,
    obs_buf: torch.Tensor,
    obs_size: torch.LongTensor,
    net_queue: mp.Queue,
    out_bufs: typing.Sequence[torch.Tensor],
    out_free: torch.BoolTensor,
    out_offset: int,
    pipes: typing.Sequence[Connection],
    all_tasks_done: ctypes.c_bool,
) -> None:
    torch.cuda.set_device(device)
    '''
    Manage net evaluations on a single device. Pop requests from the
    queue and evaluate them as inputs.
    '''

    checkpoint = load_model(model_file)
    if method == 'alpha-zero':
        net = FoldZeroNet()
        net.load_state_dict(checkpoint['net_state_dict'])
    elif method == 'deep-q':
        net = FoldQNet()
        net.load_state_dict(checkpoint['net_state_dict'][0])
    net.to(device)
    net.eval()
    with torch.no_grad():
        logger.debug('Net worker initialized (%s)', mp.current_process().name)
        while not all_tasks_done:
            # Greedily pop from the queue until it is empty, or until
            # reaching the maximum number of inputs.
            idxs = []
            while len(idxs) < MAX_CONCURRENT_EVALS:
                try:
                    idxs.append(net_queue.get_nowait())
                except queue.Empty:
                    break
            if not idxs:
                continue

            # Evaluate net on queued observations
            obs, mask = pad_sequence_with_mask(
                [obs_buf[idx, :obs_size[idx]] for idx in idxs],
                device=device,
            )
            logger.debug('Running net with %d observations (%s)',
                         len(idxs), mp.current_process().name)
            out = net(obs, mask)

            # Copy results to buffers.
            while not torch.all(out_free):
                continue
            if method == 'alpha-zero':
                out_bufs[0][:len(idxs)] = out[0]
                out_bufs[1][:len(idxs)] = out[1]
            elif method == 'deep-q':
                out_bufs[0][:len(idxs)] = out

            # Notify processes that their results are available.
            out_free[:len(idxs)].fill_(False)
            for buf_idx, proc_idx in enumerate(idxs, out_offset):
                pipes[proc_idx].send(buf_idx)

    logger.debug('Terminating net worker (%s)', mp.current_process().name)


def sim_worker(
    proc_idx: int,
    config: dict,
    iteration: int,
    evaluate: bool,
    obs_buf: torch.Tensor,
    obs_size: torch.LongTensor,
    net_queue: mp.Queue,
    pipe: Connection,
    out_buf: typing.Sequence[torch.Tensor],
    out_free: torch.BoolTensor,
    task_queue: mp.JoinableQueue,
    all_tasks_done: ctypes.c_bool,
    result_dest: str | tuple[torch.LongTensor, mp.SimpleQueue],
    barrier: Barrier,
) -> None:
    '''
    Run simulation tasks.
    '''
    def net(obs: torch.Tensor):
        # Add tensor to processing queue, wait for result
        obs_buf[:obs.size(0)] = obs
        obs_size.fill_(obs.size(0))
        net_queue.put(proc_idx)
        buf_idx = pipe.recv()
        if config['method'] == 'alpha-zero':
            p = out_buf[0][buf_idx].numpy()
            v = out_buf[1][buf_idx].item()
            out_free[buf_idx] = True
            if (np.any(np.isnan(p))):
                logger.warninging(
                    'Nan prior probability encountered! Is there a net problem?',
                )
            if (np.isnan(v)):
                logger.warning('Nan value encountered! Is there a net problem?')
            return np.exp(p.ravel()), v
        if config['method'] == 'deep-q':
            q = out_buf[0][buf_idx].detach().clone()
            out_free[buf_idx] = True
            return q
        raise ValueError('Configuration method {} not recognized'.format(config['method']))

    share_memory = config.get('simulate_shm', True)
    results = []
    tensors = []
    tensor_offsets = []

    logger.debug('Sim worker initialized (%s)', mp.current_process().name)

    while not all_tasks_done:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            continue

        # Run simulation
        if config['method'] == 'alpha-zero':
            result = run_alphazero_sim(task, net, evaluate)
            result = (
                torch.stack(result[0]),
                torch.stack(result[1]),
                result[2],
            )
            new_tensors = (result[0], result[1])
        elif config['method'] == 'deep-q':
            epsilon = config['q_epsilon_min'] + (
                (config['q_epsilon_max'] - config['q_epsilon_min'])
                * config['q_epsilon_gamma'] ** iteration
            )
            logger.debug('Current epsilon is %.3f (%s)', epsilon, mp.current_process().name)
            result = run_q_sim(task, net, evaluate, epsilon)
            # Actions must be stored as floats due to pytorch
            # TypedStorage limitation (in newer versions, tensors of
            # different dtypes can share the same UntypedStorage)
            result = (
                torch.stack(result[0]),
                torch.as_tensor(result[1], dtype=torch.float),
                result[2]
            )
            new_tensors = (result[0], result[1])

        # Append to local results
        results.append(result)

        # If enabled, collect result tensors for shm packing
        if share_memory:
            offset = result_dest[0].item()
            for tensor in new_tensors:
                tensors.append(tensor)
                tensor_offsets.append(offset)
                offset += tensor.numel()
            result_dest[0].fill_(offset)
        task_queue.task_done()
        logger.debug('Completed task (%s)', mp.current_process().name)

    # Collect results
    del net_queue
    if share_memory:
        result_tensor = pipe.recv()
        pipe.close()
        barrier.wait()
        logger.debug('Passed barrier (%s)', mp.current_process().name)
        # Move result tensors to shared storage
        torch.cat([tensor.flatten() for tensor in tensors], out=result_tensor)
        result_storage = result_tensor.storage()
        result_storage_offset = result_tensor.storage_offset()
        for tensor, offset in zip(tensors, tensor_offsets):
            tensor.set_(
                result_storage,
                result_storage_offset + offset,
                tensor.size(),
            )
        logger.debug('Sending results (%s)', mp.current_process().name)
        result_dest[1].put(results)
        barrier.wait()
    else:
        torch.save(
            results,
            os.path.join(result_dest, 'proc_{}'.format(proc_idx)),
        )
    logger.debug('Terminating sim worker (%s)', mp.current_process().name)


def batch_sim(
    config: dict,
    iteration: int,
    model_file: str,
    result_file: str | None = None,
    evaluate: bool = False,
) -> list:

    # Use the provided configuration to determine tasks
    for task in config['sequences']:
        task['file_id'] = os.path.join(config['pbd_path'], task['file_id'])
    num_sims_per_task = (
        config['num_sims_eval'] if evaluate else config['num_sims_train']
    )
    num_tasks = len(config['sequences']) * num_sims_per_task
    all_tasks = (
        task for task in config['sequences']
        for _ in range(num_sims_per_task)
    )

    # Determine CUDA availability and device count
    if torch.cuda.is_available():
        device = 'cuda'
        num_devices = torch.cuda.device_count()
    else:
        device = 'cpu'
        num_devices = 1
    logger.info('Running simulation on %s with %d device workers.', device, num_devices)

    # Initialize queues and pipes (must use spawn context for cuda workers)
    share_memory = config.get('simulate_shm', True)
    num_procs = mp.cpu_count() - num_devices
    task_queue = mp.JoinableQueue()
    barrier = mp.Barrier(num_procs + 1)
    result_queue = mp.Queue() if share_memory else None
    spawn_ctx = mp.get_context('spawn')
    net_queue = spawn_ctx.Queue()
    recv_pipes, send_pipes = zip(*(
        spawn_ctx.Pipe(False) for _ in range(num_procs)
    ))
    all_tasks_done = spawn_ctx.RawValue(ctypes.c_bool, False)

    # Initialize shared buffers
    logger.info('Using %d sim workers (%d cpus available).',
                num_procs, mp.cpu_count())
    obs_buf = torch.empty((
        num_procs,
        MAX_OBS_LEN,
        222,
    )).share_memory_()
    obs_size = torch.empty((num_procs,), dtype=torch.long).share_memory_()
    if config['method'] == 'alpha-zero':
        out_buf = (
            torch.empty((
                num_devices * MAX_CONCURRENT_EVALS,
                AZIMUTHAL_DIM * POLAR_DIM,
            )).share_memory_(),
            torch.empty((num_devices * MAX_CONCURRENT_EVALS,)).share_memory_(),
        )
    elif config['method'] == 'deep-q':
        out_buf = (
            torch.empty((
                num_devices * MAX_CONCURRENT_EVALS,
                AZIMUTHAL_DIM * POLAR_DIM,
            )).share_memory_(),
        )
    out_free = torch.ones(
        num_devices * MAX_CONCURRENT_EVALS,
        dtype=torch.bool,
    ).share_memory_()
    result_size = (
        torch.zeros((num_procs,), dtype=torch.long).share_memory_()
        if share_memory else None
    )

    # Initialize net workers (use spawn context)
    net_workers = [
        spawn_ctx.Process(
            target=net_worker,
            name='net-worker-{}'.format(i),
            args=(
                config['method'],
                model_file,
                torch.device(device, i),
                obs_buf,
                obs_size,
                net_queue,
                out_buf_i,
                out_free_i,
                i * MAX_CONCURRENT_EVALS,
                send_pipes,
                all_tasks_done,
            )
        )
        for i, out_buf_i, out_free_i in zip(
            range(num_devices),
            zip(*(x.split(MAX_CONCURRENT_EVALS) for x in out_buf)),
            out_free.split(MAX_CONCURRENT_EVALS),
        )
    ]
    for proc in net_workers:
        proc.start()
    logger.info('All net workers initialized.')

    with (
        tempfile.TemporaryDirectory() if not share_memory else nullcontext()
    ) as temp_dir:
        # Initialize sim workers (use fork context if available)
        result_dest = (
            ((result_size_i, result_queue) for result_size_i in result_size)
            if share_memory else
            (temp_dir for _ in range(num_procs))
        )
        sim_workers = [
            mp.Process(
                target=sim_worker,
                name='sim-worker-{}'.format(i),
                args=(
                    i,
                    config,
                    iteration,
                    evaluate,
                    obs_buf_i,
                    obs_size_i,
                    net_queue,
                    recv_pipe_i,
                    out_buf,
                    out_free,
                    task_queue,
                    all_tasks_done,
                    result_dest_i,
                    barrier,
                )
            )
            for i, recv_pipe_i, obs_buf_i, obs_size_i, result_dest_i in zip(
                range(num_procs),
                recv_pipes,
                obs_buf,
                obs_size,
                result_dest,
            )
        ]
        for proc in sim_workers:
            proc.start()
        logger.info('All sim workers initialized.')

        # Run all sim tasks
        for task in all_tasks:
            task_queue.put(task)
        logger.info('All %d simulation tasks queued.', num_tasks)
        task_queue.join()
        all_tasks_done.value = True
        logger.info('All simulation tasks complete.')

        # Free net workers.
        for proc in net_workers:
            proc.join()
            proc.close()
        logger.info('All net workers released.')
        del net_queue

        results = []
        if share_memory:
            # Collect results into a single list using a contiguous block of
            # shared memory.
            logger.debug('Result sizes: %s', repr(result_size))
            logger.debug('Total number of elements: %s', result_size.sum())
            results_storage = torch.empty(result_size.sum()).share_memory_()
            for storage, pipe in zip(
                results_storage.split(result_size.tolist()),
                send_pipes,
            ):
                pipe.send(storage)
                pipe.close()
            logger.debug('Sent all storage buffers')
            barrier.wait()
            # for pipe in send_pipes:
            #     pipe.close()
            logger.debug('Closed all pipe connections')
            for _ in range(num_procs):
                results.extend(result_queue.get())
            barrier.wait()
        else:
            for file in os.listdir(temp_dir):
                proc_results = torch.load(os.path.join(temp_dir, file))
                results.extend(proc_results)
        if len(results) != num_tasks:
            logger.warning(
                'Number of simulation results (%d) does not match number of tasks (%d)!', len(results), num_tasks)

        # Free sim workers.
        for proc in sim_workers:
            proc.join()
            proc.close()
        logger.info('All sim workers released.')

    # Optionally save results to file
    if result_file is not None:
        if not os.path.exists(os.path.dirname(result_file)):
            os.makedirs(os.path.dirname(result_file))
        torch.save(
            (results, results_storage.storage())
                if config.get('dataset_shm', False) else
                results,
            result_file,
        )
        logger.info('All simulation results saved to %s.', result_file)
    return results
