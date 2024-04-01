# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from models.memory_modules import TreeMemory, VanillaMemory
import torch.nn as nn


class Processor(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm_first):
        super(Processor, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.norm_first = norm_first

    def forward(self, layer_data):
        raise NotImplementedError


class TreeProcessor(Processor):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        norm_first,
        branch_factor,
        num_aggregation_layers,
        bptt,
        aggregator_type,
    ):
        super(TreeProcessor, self).__init__(
            d_model, nhead, dim_feedforward, dropout, norm_first
        )
        self.branch_factor = branch_factor
        self.num_aggregation_layers = num_aggregation_layers

        self.memory_block = TreeMemory(
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            self.dropout,
            self.norm_first,
            self.branch_factor,
            self.num_aggregation_layers,
            bptt,
            aggregator_type=aggregator_type,
        )

    def forward(self, layer_data):
        self.memory_block.setup_data(layer_data)
        return self.memory_block

    def reset(self):
        self.memory_block.reset()


class VanillaProcessor(Processor):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm_first):
        super(VanillaProcessor, self).__init__(
            d_model, nhead, dim_feedforward, dropout, norm_first
        )

        self.memory_block = VanillaMemory(
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            self.dropout,
            self.norm_first,
        )

    def forward(self, layer_data):
        self.memory_block.setup_data(layer_data)
        return self.memory_block

    def reset(self):
        self.memory_block.reset()
