# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from models.attention_modules import *
import torch.nn as nn
from torch import Tensor

from torch import nn
from models.misc import _get_clones


##### Encoder Layers


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        pass

    def forward(self, src: Tensor):
        """
        Arguments:
            src: [B, N, D] Tensor
        Returns:
            ret: [B, N, D] Tensor
        """
        raise NotImplementedError


class TransformerEncoderLayer(EncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        norm_first: bool = True,
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.d_model = d_model

        assert self.d_model % nhead == 0

        if norm_first:
            Norm = PreNorm
        else:
            Norm = PostNorm

        self.dataset_self_attn = Norm(
            d_model,
            Attention(d_model, nhead=nhead, dim_head=d_model // nhead, dropout=dropout),
        )
        self.ff = Norm(
            d_model,
            FeedForward(d_model, dim_feedforward=dim_feedforward, dropout=dropout),
        )

    def forward(self, context, src_mask=None):
        x = context
        x = self.dataset_self_attn(
            x, key=x, value=x, src_mask=src_mask, return_info=False
        )
        x = self.ff(x)
        return x


class NullEncoderLayer(EncoderLayer):
    def __init__(self):
        super(NullEncoderLayer, self).__init__()

    def forward(self, src: Tensor):
        out = src
        return out


##### Encoder


class TransformerEncoder(
    nn.Module
):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        output = src
        for layer in self.layers:
            if src_mask is not None:
                output = layer(output, src_mask=src_mask)
            else:
                output = layer(output)
        return output


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src: Tensor):
        output = src

        for layer in self.layers:
            output = layer(output)
        return output
