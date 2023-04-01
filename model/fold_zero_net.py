'''
This file defines the 2-headed network used to approximate policy and
value for a given input state.

Created on 2/28/2023 by Steven Laverty (lavers@rpi.edu)
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


class FoldZeroNet(nn.Module):
    '''
    Take a sequence of past (already placed groups) and future (unplaced
    groups) and generate a policy / value estimate for the current
    folding state.

    Inputs:
    obs -- a tensor of shape (L, N, in_dim) to calculate policy and value
        estimates for. Observations of length less than L are expected to
        be padded with nan.

    Outputs:
    p -- tensor of policy log probabilities, one for each action, for
        each observation.
    v -- tensor of value predictions for each observation.
    '''

    # Inputs
    in_dim = 222

    # Main trunk (attention)
    embed_dim = 256
    num_heads = 8
    trunk_fc_dim = 1024
    dropout = 0.1
    num_attention_layers = 16

    # Policy head
    policy_out_dim = 288

    # Fully-connected output
    value_fc_dim = 256

    def __init__(self) -> None:
        super().__init__()
        
        # Special tokens
        self.policy_token = nn.Parameter(torch.randn(self.embed_dim))
        self.value_token = nn.Parameter(torch.randn(self.embed_dim))

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

        # Policy head
        self.policy_attn = nn.MultiheadAttention(
            self.embed_dim,
            self.num_heads,
            self.dropout,
        )
        self.policy_out = nn.Linear(self.embed_dim, self.policy_out_dim)
        self.policy_logsoftmax = nn.LogSoftmax(-1)

        # Value head
        self.value_attn = nn.MultiheadAttention(
            self.embed_dim,
            self.num_heads,
            self.dropout,
        )
        self.value_fc = nn.Linear(self.embed_dim, self.value_fc_dim)
        self.value_relu = nn.ReLU()
        self.value_out = nn.Linear(self.value_fc_dim, 1)
        self.value_softplus = nn.Softplus()
    
    def forward(self, obs: torch.Tensor, key_padding_mask: torch.Tensor):
        # Broadcast policy and value tokens to batch size (1, N, embed_dim)
        policy_token, value_token, _ = torch.broadcast_tensors(
            self.policy_token.view(1, 1, self.embed_dim),
            self.value_token.view(1, 1, self.embed_dim),
            torch.empty((1, obs.size(1), 1)),
        )
        logger.debug('Tokens require grad: %s %s', policy_token.requires_grad, value_token.requires_grad)

        # Encode inputs (including policy and value tokens)
        x = torch.cat((
            policy_token,
            value_token,
            self.encoder(obs),
        ))
        logger.debug('Input requires grad: %s', x.requires_grad)

        # Extend key padding mask to include the policy and value tokens
        key_padding_mask = torch.cat(
            (
                key_padding_mask.new_zeros((key_padding_mask.size(0), 2)),
                key_padding_mask,
            ),
            dim=1,
        )

        # Apply self-attention trunk
        for attn in self.attn_trunk:
            x = attn(x, key_padding_mask)

        # Apply policy head
        p, _ = self.policy_attn(
            x[0].unsqueeze(0),
            x,
            x,
            key_padding_mask,
            need_weights=False,
        )
        p = self.policy_logsoftmax(self.policy_out(p.squeeze(0)))

        # Apply value head
        v, _ = self.value_attn(
            x[1].unsqueeze(0),
            x,
            x,
            key_padding_mask,
            need_weights=False,
        )
        v = self.value_relu(self.value_fc(v.squeeze(0)))
        v = -self.value_softplus(self.value_out(v).squeeze(1))

        return p, v
