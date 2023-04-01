import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import quaternion

import fold_sim
from fold_sim.envs import group_fold
from fold_sim.wrappers import AngleActions, ObservationDict, NormalizedRewards, TransformerInputObs

env = gym.make('fold_sim/ResGroupFoldDiscrete-v0')

print(env.observation_space)
print(env.action_space)

if False:
    env = ObservationDict(env)
    print('Using observation wrapper:')
    print(env.observation_space)
    print(env.action_space)

if False:
    env = AngleActions(env)
    print('Using action wrapper:')
    print(env.observation_space)
    print(env.action_space)

if True:
    env = TransformerInputObs(env)
    print('Using transformer input observation wrapper:')
    print(env.observation_space)

if True:
    env = NormalizedRewards(env)
    print('Using normalized rewards:')

if False:
    '''
    Plot the action space
    '''
    action_frames = np.ravel(env._action_to_quat[:, :, 0])
    basis = np.eye(3)
    local_basis = quaternion.rotate_vectors(action_frames, basis)
    local_origin = np.copy(local_basis[:, 0])
    local_basis *= 0.1
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.quiver(-1, 0, 0, 1, 0, 0, color='k')
    ax.scatter(0, 0, 0, color='k')
    ax.scatter(*local_origin.T, color='k', marker='.')
    ax.quiver(*local_origin.T, *local_basis[:, 0].T, color='r')
    ax.quiver(*local_origin.T, *local_basis[:, 1].T, color='b')
    ax.quiver(*local_origin.T, *local_basis[:, 2].T, color='g')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('Action space')
    plt.show()

'''
Test reset
'''
obs, info = env.reset(
    seed=1,
    options={
        'file_id': '1dm7.cif',
        # 'file_id': '2igd.cif',
        'chain_id': 'A',
    },
)

# print(obs.shape)
# print(info)

if True:
    true_action_rotations = info['true_group_rotations'][:-1].swapaxes(-1, -2) @ info['true_group_rotations'][1:]
    discrete_action_rotations = quaternion.as_rotation_matrix(
        group_fold.ACTION_TO_QUAT
    )
    dot_product = (
        true_action_rotations[:, None] * discrete_action_rotations[None]
    ).sum(axis=-1).sum(axis=-1)
    optimal_policy = np.argmax(dot_product, axis=-1)

    for action in optimal_policy:
        obs, reward, done, _, info = env.step(action)
    assert done
    if True:
        true_pdb, sim_pdb = env.unwrapped.export_pdb()
        with open('test_true.pdb', 'w') as f:
            f.write(true_pdb)
        with open('test_sim.pdb', 'w') as f:
            f.write(sim_pdb)
    print('Optimal policy: {}'.format(optimal_policy))
    print('Optimal policy loss: {}'.format(reward))

if False:
    '''
    Test run
    '''
    done = False
    while not done:
        obs, reward, done, _, info = env.step(
            # (0, 0) # straight
            # (1, 0) # rotation
            12
            # (0, 1) # curve
            # (1, 1) # helix
            # (6, 11) # double helix
            # (23, 11) # a different helix
            # (23 * 12) + 11 # ^ flat actions
            # (np.random.randint(24), np.random.randint(12)) # random
            # np.random.randint(24 * 12) # ^ flat actions
        )
        if (env.state.idx == 32):
            saved_state = env.unwrapped.state
            # print(obs[0][:, :10])
        # print(obs)
        # print(loss)
        # print(info)

    print(obs.shape)
    # print(np.min(obs[0]), np.max(obs[0]))
    print(reward)

    if False:
        env.unwrapped.state = saved_state
        print(env.step((0, 0, 0))[0])

if True:
    '''
    Plot the true structure
    '''
    n_plot = 20
    n_plot = len(info['true_group_locations']) - 1

    if True:
        first_rotation = info['true_group_rotations'][0]
        info['true_group_locations'] -= info['true_group_locations'][0][None]
        info['true_group_locations'] = (first_rotation.T[None] @ info['true_group_locations'][..., None]).squeeze(-1)
        info['true_group_rotations'] = first_rotation.T[None] @ info['true_group_rotations']

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(
        *info['true_group_locations'][:n_plot + 1].T,
        color='k',
        marker='.',
    )

    if True:
        local_basis = 0.3 * np.max(np.linalg.norm(
            info['true_group_locations'][1:n_plot + 1]
            - info['true_group_locations'][:n_plot],
            axis=-1,
        )) * info['true_group_rotations'][1:n_plot]
        ax.quiver(*info['true_group_locations'][1:n_plot].T,
                  *local_basis[..., 0].T, color='r')
        ax.quiver(*info['true_group_locations'][1:n_plot].T,
                  *local_basis[..., 1].T, color='b')
        ax.quiver(*info['true_group_locations'][1:n_plot].T,
                  *local_basis[..., 2].T, color='g')

    ax.set_box_aspect(tuple(
        np.ptp(x) for x in info['true_group_locations'][:n_plot + 1].T
    ))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('True structure')
    plt.show()

if True:
    '''
    Plot the simulation
    '''
    n_plot = 20
    n_plot = len(info['sim_group_locations']) - 1

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(
        *info['sim_group_locations'][:n_plot + 1].T,
        color='k',
        marker='.',
    )

    if True:
        local_basis = quaternion.rotate_vectors(
            info['sim_group_rotations'][1:n_plot],
            0.3 * np.max(np.linalg.norm(
                info['sim_group_locations'][1:n_plot + 1]
                - info['sim_group_locations'][:n_plot],
                axis=-1,
            )) * np.eye(3)
        )
        ax.quiver(*info['sim_group_locations'][1:n_plot].T,
                  *local_basis[:, 0].T, color='r')
        ax.quiver(*info['sim_group_locations'][1:n_plot].T,
                  *local_basis[:, 1].T, color='b')
        ax.quiver(*info['sim_group_locations'][1:n_plot].T,
                  *local_basis[:, 2].T, color='g')

    ax.set_box_aspect(tuple(
        np.fmax(np.ptp(x), 1e-1) for x in info['sim_group_locations'][:n_plot + 1].T
    ))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('Sample run')
    plt.show()
