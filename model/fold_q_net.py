'''
This file defines the artificial neural network used to approximate
Q action-values for a given input state. For use in deep Q learning
RL algorithm.

Created on 3/15/2023 by Steven Laverty (lavers@rpi.edu)
'''

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class SelfAttentionLayer(nn.Module):
    '''
    Compute self-attention (optionally with dropout), followed by
    a fully-connected linear layer.

    Inputs:
    x -- a tensor of size (L, N, E) where L is the length of the
        sequence, N is the batch size, and E is the embedding dimension.
    key_padding_mask -- a binary tensor of shape (N, L) where True indicates
        that this is a padded addition to the sequence and it should be
        ignored during self-attention.
    
    Outputs:
    x -- a tensor of size (L, N, E) with self attention
    '''

    def __init__(
        self,
        embed_dim,
        num_heads,
        fc_dim,
        dropout=0.1,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        k_dim = None,
        v_dim = None,
    ) -> None:
        super().__init__()

        # Multi-headed attention
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            k_dim,
            v_dim,
        )

        # Feed-forward layer
        self.fc1 = nn.Linear(embed_dim, fc_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_dim, embed_dim)


    def forward(self, x, key_padding_mask=None):
        # Self-attention
        y, _ = self.attn(x, x, x, key_padding_mask, need_weights=False)

        # Residual connection
        x = x + y

        # Fully-connected layers
        y = self.fc2(self.relu(self.fc1(x)))

        # Residual connection
        x = x + y

        return x


class FoldQNet(nn.Module):
    '''
    Take a sequence of past (already placed groups) and future (unplaced
    groups) and generate a policy / value estimate for the current
    folding state.

    Inputs:
    obs -- a tensor of shape (L, N, in_dim) to calculate policy and value
        estimates for. Observations of length less than L are expected to
        be padded with nan.
    mask -- a binary tensor of shape (N, L) where True indicates
        that this is a padded addition to the sequence and it should be
        ignored during self-attention.

    Outputs:
    q -- tensor of q action-values, one for each action.
    '''

    # Inputs
    in_dim = 222

    # Main trunk (attention)
    embed_dim = 256
    num_heads = 8
    trunk_fc_dim = 1024
    dropout = 0.1
    num_attention_layers = 16

    # Q-value head
    q_fc_dim = 1024
    num_actions = 288


    def __init__(self) -> None:
        super().__init__()
        
        # Special tokens
        self.q_token = nn.Parameter(torch.randn(self.embed_dim))

        # input embedding
        self.encoder = nn.Linear(self.in_dim, self.embed_dim)

        # Self-attention trunk
        self.attn_trunk = nn.ModuleList(
            SelfAttentionLayer(
                self.embed_dim,
                self.num_heads,
                self.trunk_fc_dim,
                self.dropout,
            )
            for _ in range(self.num_attention_layers)
        )

        # Q value head
        self.q_attn = nn.MultiheadAttention(
            self.embed_dim,
            self.num_heads,
            self.dropout,
        )
        self.q_fc1 = nn.Linear(self.embed_dim, self.q_fc_dim)
        self.q_relu = nn.ReLU()
        self.q_fc2 = nn.Linear(self.q_fc_dim, self.embed_dim)
        self.q_out = nn.Linear(self.embed_dim, self.num_actions)

    
    def forward(self, obs: torch.Tensor, key_padding_mask: torch.Tensor):
        # Expand q value token to batch size (1, N, embed_dim)
        q_token = self.q_token.expand((1, obs.size(1), -1))
        logger.debug('Token requires grad: %s', q_token.requires_grad)

        # Encode inputs (including q value token)
        x = torch.cat((
            q_token,
            self.encoder(obs),
        ))
        logger.debug('Input requires grad: %s', x.requires_grad)

        # Extend key padding mask to include the q value token
        key_padding_mask = torch.cat(
            (
                key_padding_mask.new_zeros((key_padding_mask.size(0), 1)),
                key_padding_mask,
            ),
            dim=1,
        )

        # Apply self-attention trunk
        for attn in self.attn_trunk:
            x = attn(x, key_padding_mask)

        # Apply q value head
        q, _ = self.q_attn(
            x[0].unsqueeze(0),
            x,
            x,
            key_padding_mask,
            need_weights=False,
        )
        q = self.q_fc2(self.q_relu(self.q_fc1(q.squeeze(0))))
        q = self.q_out(q)

        return q
