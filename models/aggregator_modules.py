# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch.nn as nn

from torch import nn
from models.attention_modules import *
from models.encoder_modules import TransformerEncoder, TransformerEncoderLayer


class Aggregator(nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, data, node_mask=None):
        raise NotImplementedError


class TransformerAggregator(Aggregator):
    def __init__(
        self,
        num_aggregation_layers,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        norm_first,
        bptt=True,
    ):
        super(TransformerAggregator, self).__init__()

        layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=norm_first,
        )

        self.encoder = TransformerEncoder(layer, num_aggregation_layers)
        self.bptt = bptt

    def forward(self, data, node_mask=None):
        """
        Args:
            - data: [B..., N, D]
            - node_mask: [B..., N, 1] - describes which nodes are padding nodes
        Returns:
            - [B..., 1, D] - aggregation of the embeddings
        """
        if not self.bptt:
            data = data.detach()

        if node_mask is None:
            return self.encoder(data).mean(1, keepdim=True)
        else:
            mask = node_mask @ node_mask.transpose(-2, -1)
            N_real = node_mask.sum(-2).unsqueeze(-1) + 1e-9  # [B..., 1, 1]

            embeddings = self.encoder(data, mask) * node_mask
            return embeddings.sum(1, keepdim=True) / N_real
