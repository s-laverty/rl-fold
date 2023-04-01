import typing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

GRU_HIDDEN_DIM = 512
GRU_STACKS = 20

class ValueNet(nn.Module):
    '''
    Take a sequence of placed residue groups, estimate the log loss.

    Inputs
    --
    x -- zero-padded tensor with dimensions (R, B, d_obs) where B is the
    batch size, R is the maximum number of groups over all
    observations in the batch, and d_obs is the size of an
    observation vector for a single group.

    lengths -- tensor of size (B) with data type torch.64 corresponding
        to the number of groups in each observation of the batch.

    Outputs
    --
    v -- tensor of value predictions for each observation.
    h -- the final hidden state representation of the GRU stack.
    '''

    in_dim = 187
    fc2_dim = 256

    class ResGRU(nn.Module):
        '''
        Apply a GRU layer to a given input, then output the sum of
        the GRU result with the original input (residual connection)

        Inputs
        --
        x -- zero-padded tensor with dimensions (R, B, d_obs) where B is the
        batch size, R is the maximum number of groups over all
        observations in the batch, and d_obs is the size of an
        observation vector for a single group.

        lengths -- tensor of size (B) with data type torch.64 corresponding
            to the number of groups in each observation of the batch.

        Outputs
        --
        x -- tensor of value predictions for each observation.
        h -- the final hidden state representations of the GRU stack.
        '''

        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_dim=GRU_HIDDEN_DIM,
                hidden_dim=GRU_HIDDEN_DIM,
                bias=True,
            )
            self.dropout = nn.Dropout(p=0.4)

        def forward(
            self,
            x_seq: rnn.PackedSequence,
        ) -> typing.Tuple[rnn.PackedSequence, torch.Tensor]:
            # Calculate GRU layer
            y_seq, h = self.gru(x_seq)

            # Dropout layer
            y = self.dropout(y_seq.data)

            # Add residual connection
            x = x_seq.data + y

            # Re-pack (PackedSequence HACK)
            x_seq = rnn.PackedSequence(
                x,
                x_seq.batch_dims,
                x_seq.sorted_indices,
                x_seq.unsorted_indices
            )
            return x_seq, h

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(self.in_dim, GRU_HIDDEN_DIM)
        self.dropout1 = nn.Dropout(p=0.1)
        self.gru_stack = nn.ModuleList(
            self.ResGRU() for _ in range(GRU_STACKS)
        )
        self.fc2 = nn.Linear(GRU_HIDDEN_DIM, self.fc2_dim)
        self.relu2 = nn.LeakyReLU(0.01)
        self.fc3 = nn.Linear(self.fc2_dim, 1)

    def forward(
        self,
        x_list: typing.List[torch.Tensor],
        lengths: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # Pack sequences
        x_seq = rnn.pack_sequence(x_list, enforce_sorted=False)

        # Fully connected conversion to hidden dimension
        x = self.fc1(x_seq.data)
        x = self.dropout1(x)

        # Re-pack (PackedSequence HACK)
        x_seq = rnn.PackedSequence(
            x,
            x_seq.batch_dims,
            x_seq.sorted_indices,
            x_seq.unsorted_indices
        )

        # Main GRU stack (with residual connections)
        h = []
        for gru in self.gru_stack:
            x, h_i = gru(x)
            h.append(h_i)
        h = torch.cat(h)

        # Calculate values from the final outputs of the GRU stack
        x = x[lengths - 1, np.arange(x.size(1))]
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = torch.squeeze(x, 1)
        v = -torch.exp(x)
        return v, h


class PolicyNet(nn.Module):
    '''
    Take a sequence of future (unplaced) residue groups along with the
    hidden state representation of the current structure, estimate the
    action probabilities for every action.

    Inputs
    --
    x -- zero-padded tensor with dimensions (R, B, d_obs) where B is the
    batch size, R is the maximum number of groups over all
    observations in the batch, and d_obs is the size of an
    observation vector for a single group.

    lengths -- tensor of size (B) with data type torch.64 corresponding
        to the number of groups in each observation of the batch.

    h -- final hidden state representation from the StructureValueNet,
        used for every iteration in the PolicyNet.

    Output
    --
    p -- tensor of policy log probabilities, one for each action.
    '''

    
    class ResGRU(nn.Module):
        '''
        Apply a GRU layer to a given input, then output the sum of
        the GRU result with the original input (residual connection)

        Inputs
        --
        x -- zero-padded tensor with dimensions (R, B, d_obs) where B is the
        batch size, R is the maximum number of groups over all
        observations in the batch, and d_obs is the size of an
        observation vector for a single group.

        lengths -- tensor of size (B) with data type torch.64 corresponding
            to the number of groups in each observation of the batch.
        
        h_struct -- the corresponding hidden state representation from
            the StructureValueNet.

        Outputs
        --
        x -- tensor of value predictions for each observation.
        '''

        def __init__(self) -> None:
            super().__init__()

            self.gru = nn.GRU(
                input_dim=2 * GRU_HIDDEN_DIM,
                hidden_dim=GRU_HIDDEN_DIM,
                bias=True,
                dropout=0.4,
            )

        def forward(
            self,
            x: torch.Tensor,
            lengths: torch.Tensor,
            h_struct: torch.Tensor,
        ) -> torch.Tensor:
            # Calculate GRU layer
            # Concatenating at every time step requires stupid outer-
            # product broadcasting.
            y = torch.cat(
                (
                    x,
                    h_struct * torch.ones(
                        (x.size(0), 1, 1),
                        dtype=h_struct.dtype,
                        layout=h_struct.layout,
                        device=h_struct.device,
                    ),
                ),
                -1,
            )
            y = rnn.pack_padded_sequence(
                y,
                lengths,
                enforce_sorted=False,
            )

            # Add residual connection
            y, _ = self.gru(y)
            y, _ = rnn.pad_packed_sequence(y)
            x = x + y
            return x

    in_dim = 179
    fc2_dim = 256
    out_dim = 288

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(self.in_dim, GRU_HIDDEN_DIM)
        self.dropout1 = nn.Dropout(p=0.1)
        self.gru_stack = nn.ModuleList(
            self.ResGRU() for _ in range(GRU_STACKS)
        )
        self.fc2 = nn.Linear(GRU_HIDDEN_DIM, self.fc2_dim)
        self.relu2 = nn.LeakyReLU(0.01)
        self.fc3 = nn.Linear(self.fc2_dim, self.out_dim)
        self.logsoftmax3 = nn.LogSoftmax(1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        h_struct: torch.Tensor,
    ) -> torch.Tensor:
        # Fully connected conversion to hidden dimension
        x = self.fc1(x)
        x = self.dropout1(x)

        # Main GRU stack (with residual connections)
        h_struct = h_struct.split(1)
        for gru, h_i in zip(self.gru_stack, h_struct):
            x = gru(x, lengths, h_i)

        # Calculate policy from the final outputs of the GRU stack
        x = x[lengths - 1, np.arange(x.size(1))]
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        p = self.logsoftmax3(x)
        return p


class OldFoldNet(nn.Module):
    '''
    Take a sequence of past (already placed groups) and future (unplaced
    groups) and generate a policy / value estimate for the current
    folding state.

    Inputs:
    obs -- a list of observations to calculate policy and value
        estimates for.

    Outputs:
    p -- tensor of policy log probabilities, one for each action, for
        each observation.
    v -- tensor of value predictions for each observation.
    '''

    def __init__(self) -> None:
        super().__init__()
        self.value_net = ValueNet()
        self.policy_net = PolicyNet()

    def forward(
        self,
        obs: typing.List[typing.Tuple[torch.Tensor, torch.Tensor]],
        
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # Get input lengths
        past_obs, future_obs = map(list, zip(*obs))
        past_lengths = torch.as_tensor(
            [obs.size(0) for obs in past_obs]
        )
        future_lengths = torch.as_tensor([obs.size(0) for obs in future_obs])

        # Calculate value estimates.
        v, h_struct = self.value_net(rnn.pad_sequence(past_obs), past_lengths)

        # Calculate policy -- only non-terminal states are valid inputs.
        p = torch.zeros(
            (v.size(0), PolicyNet.out_dim),
            dtype=v.dtype,
            layout=v.layout,
            device=v.device,
            requires_grad=True
        )
        non_terminal = future_lengths > 0
        if torch.any(non_terminal):
            p_mask = self.policy_net(
                rnn.pad_sequence([
                    obs for obs, is_nt in zip(future_obs, non_terminal)
                    if is_nt
                ]),
                future_lengths[non_terminal],
                h_struct[:, non_terminal],
            )
            p[non_terminal] = p_mask

        # Return policy (logits) and value estimates.
        return p, v
