'''
This file implements the AlphaZero Monte-Carlo tree search algorithm
for generating a nondeterministic policy.

Adapted from the github repository: https://github.com/plkmo/AlphaZero_Connect4

Created on 2/15/2023 by Steven Laverty (lavers@rpi.edu)
'''

from __future__ import annotations

import logging
import typing

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class UCTNode():
    '''
    Upper-confidence bound tree node representing a specific environment
    state in a Monte-Carlo tree search.
    '''

    c_puct: float
    '''
    The upper-confidence bound coefficient. Higher values
    encourage exploration.
    '''

    bias: float
    '''
    The initial assumed Q value for all actions.
    '''

    state: object
    '''
    The environment state this node represents.
    '''

    parent: UCTNode | None
    '''
    The parent node that yielded this node's state. If this is the root
    node of the MCTS, then parent is None.
    '''

    action_idx: int | None
    '''
    The index of the action that yielded this node's state. If this is
    the root node of the MCTS, then action_idx is None.
    '''

    actions: np.ndarray
    '''
    A list of actions available to the agent from this node's state.
    '''

    action_priors: np.ndarray
    '''
    The prior probabilities for every action available to the agent from
    this node's state.
    '''

    action_Q_values: np.ndarray
    '''
    The average value over all subtree evaluations for every action
    available to the agent from this node's state.
    '''

    children: dict[int, UCTNode]
    '''
    The set of expanded child nodes that this node's state can yield.
    '''

    child_number_visits: np.ndarray
    '''
    The total number of visits for every action available to the agent
    from this node's state.
    '''

    @property
    def number_visits(self) -> int:
        '''
        The total number of visits for this node's state.
        '''
        if self.parent is None:
            return self._number_visits
        return self.parent.child_number_visits[self.action_idx]

    @number_visits.setter
    def number_visits(self, value: int):
        if self.parent is None:
            self._number_visits = value
        else:
            self.parent.child_number_visits[self.action_idx] = value

    @property
    def Q_value(self) -> int:
        '''
        The average accumulated value for this node's state.
        '''
        if self.parent is None:
            return self._Q_value
        return self.parent.child_number_visits[self.action_idx]

    @Q_value.setter
    def Q_value(self, value: int):
        if self.parent is None:
            self._Q_value = value
        else:
            self.parent.action_Q_values[self.action_idx] = value

    def __init__(
        self,
        state: object,
        actions: np.ndarray,
        action_priors: np.ndarray,
        parent: UCTNode | None = None,
        action_idx: int | None = None,
        *,
        c_puct: float = 1.0,
        bias: float = 0.0,
        c_noise: float = 0.25,
        dirichlet_alpha: float = 0.03,
    ) -> None:
        '''
        Required arguments
        --
        state -- the environment state to be represented by this node.

        actions -- the vector of valid actions that can be taken from
            this state. For terminal states, this is a zero-length
            array.

        priors -- the vector of prior probabilities for each action. For
            terminal states, this should be a zero-length array.

        Optional arguments
        --
        parent -- the parent node whose environment state yielded this
            one.

        action_idx -- the action taken the parent state to yield this one.

        Keyword arguments
        --
        c_puct -- the upper-confidence bound coefficient.

        c_noise -- the coefficient used for adding dirichlet noise to
            the priors.

        dirichlet_alpha -- the alpha value used for adding dirichlet
            noise to the priors.
        '''

        # Save constants for future calculations
        self.c_puct = c_puct
        self.bias = bias

        # Initialize this node
        self.state = state
        self.parent = parent
        self.action_idx = action_idx

        # Special initialization for root node
        if self.parent is None:
            self.number_visits = 0
            self.Q_value = 0.0

        # Initialize action priors and Q values
        self.actions = actions
        self.action_priors = action_priors.astype(np.float64)
        self.action_Q_values = np.zeros_like(self.actions, dtype=np.float64) + self.bias
        self.children = {}
        self.child_number_visits = np.zeros_like(self.actions, dtype=np.int64)

        # Add optional dirichlet noise
        if c_noise > 0.0:
            self.add_dirichlet_noise(c_noise, dirichlet_alpha)

    def add_dirichlet_noise(self, c_noise, dirichlet_alpha) -> None:
        '''
        Add dirichlet noise to the action priors.
        '''
        self.action_priors = (
            c_noise * np.random.dirichlet(
                np.zeros(len(self.action_priors)) + dirichlet_alpha,
            )
            + (1 - c_noise) * self.action_priors
        )
    
    def reset(self) -> None:
        '''
        Recursively reset this node and all of its children's visit
        counts / Q values to zero.
        '''
        self.action_Q_values.fill(self.bias)
        self.child_number_visits.fill(0)
        for child in self.children.values():
            child.reset()

    def action_U(self) -> np.ndarray:
        '''
        Get the upper-confidence bound for all valid actions based on
        child priors.
        '''
        return (
            self.c_puct * self.action_priors
            * np.sqrt(self.number_visits)
            / (1 + self.child_number_visits)
        )

    def best_action(self) -> int | None:
        '''
        Get the index of the action with the highest Q + U value. If
        this is a leaf node (no valid actions), return None.
        '''
        if len(self.actions) == 0:
            return None
        return np.argmax(self.action_Q_values + self.action_U())

    def select_leaf(self) -> tuple[UCTNode, int | None]:
        '''
        Recursively select child nodes with the highest Q + U values
        until reaching one that hasn't been visited. Return the
        corresponding state-action pair (for leaf states that are
        expanded already, the action will be None).
        '''
        node = self
        action_idx = node.best_action()
        while node.number_visits > 0 and action_idx in node.children:
            node = node.children[action_idx]
            action_idx = node.best_action()
        return node, action_idx if node.number_visits > 0 else None

    def add_child(
        self,
        action: int,
        child_state: object,
        child_actions: np.ndarray,
        child_action_priors: np.ndarray,
        *,
        c_noise: float = 0.0,
        dirichlet_alpha: float = 128.0,
    ) -> UCTNode:
        '''
        Add a child node to this node (the child's state is the result
        of taking an action from this node's state).
        '''
        self.children[action] = UCTNode(
            child_state,
            child_actions,
            child_action_priors,
            self,
            action,
            c_puct=self.c_puct,
            c_noise=c_noise,
            dirichlet_alpha=dirichlet_alpha,
        )

        return self.children[action]

    def backup(self, value_estimate: float | None = None) -> None:
        '''
        Update this node and its parents with an estimated value.

        When this method is called for a subsequent time, the
        value_estimate parameter can be omitted, in which case the
        previous value_estimate will be used again.
        '''
        if value_estimate is None:
            if not hasattr(self, '_value_estimate'):
                raise ValueError(
                    'value_estimate can only be None if this method has been called at least once before for this node.')
            value_estimate = self._value_estimate
        else:
            self._value_estimate = value_estimate

        # Update visits and Q values for all parent nodes up to the root.
        node = self
        while node is not None:
            node.number_visits += 1
            node.Q_value += (
                (value_estimate - node.Q_value)
                / node.number_visits
            )
            node = node.parent
    
    def make_root(
        self,
        c_noise: float = 0.25,
        dirichlet_alpha: float = 0.03,
    ) -> None:
        '''
        Sever this node's ties to its parents and set it as the new
        root node.
        '''

        # The number of visits is currently stored in the parent, so it
        # must be copied.
        self.parent = self.action_idx = None
        self.number_visits = 0
        self.Q_value = 0.0
        self.reset()

        # Add optional dirichlet noise
        if c_noise > 0.0:
            self.add_dirichlet_noise(c_noise, dirichlet_alpha)


def UCT_search(
    env: gym.Env,
    net: typing.Callable[[object], tuple[np.ndarray, float]],
    actions: typing.Callable[[object], np.ndarray],
    obs: object,
    num_reads: int = 1600,
    root: UCTNode | None = None,
    *,
    bias: float = 0.0,
) -> UCTNode:
    '''
    Perform a Monte-Carlo tree search at a given state to determine the
    policy.

    Required arguments
    --
    env -- the gymnasium environment to perform a Monte-Carlo tree
        search on. The environment's internal state will be modified.

    net -- the neural network responsible for generating a policy and
        a value estimate from an observation.

    actions -- a function that determines which actions are available
        for a given environment observation.
    
    obs -- the initial observation from the root state.

    num_reads -- the number

    Optional arguments
    --
    temp -- the temperature parameter for calculating policy from visit
        counts.

    Returns
    --
    tuple[ndarray, ndarray] -- a list of actions to choose from and a
        corresponding probability vector.
    '''
    # Initialize root node.
    if root is None:
        valid_actions = actions(obs)
        action_priors, value_estimate = net(obs)
        root = UCTNode(
            env.unwrapped.state,
            valid_actions,
            action_priors[valid_actions],
            bias=bias,
        )
        root.backup(value_estimate)
    else:
        root.make_root()

    # Perform tree search.
    while root.number_visits < num_reads:
        leaf, action = root.select_leaf()
        if action is not None:
            env.unwrapped.state = leaf.state
            obs, _, terminated, _, _ = env.step(leaf.actions[action])
            valid_actions = (
                np.empty(0, dtype=np.int64) if terminated
                else actions(obs)
            )
            action_priors, value_estimate = net(obs)
            leaf = leaf.add_child(
                action,
                env.unwrapped.state,
                valid_actions,
                np.empty(0) if terminated
                else action_priors[valid_actions],
            )
        leaf.backup(None if action is None else value_estimate)

    # Reset the environment.
    env.unwrapped.state = root.state

    # Return the root of the search tree
    return root
