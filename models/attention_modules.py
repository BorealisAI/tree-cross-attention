# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2021, Phil Wang
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Attention code is based on the Perceiver (https://arxiv.org/abs/2103.03206) implementation
# from https://github.com/lucidrains/Perceiver-pytorch by Phil Wang
####################################################################################

import torch
import torch.nn as nn
from torch import Tensor

from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs) + x


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.norm(x + self.fn(x, **kwargs))


class AttNormClass(nn.Module):
    """
    Special Norm class to handle the multiple outputs required by the below Attention class.
    In addition, allows the to_out function of Attention
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.nhead = self.fn.nhead

    def forward(self, x, **kwargs):
        raise NotImplementedError

    def to_out(self, x, **kwargs):
        return self.fn.to_out(x, **kwargs)


class AttPreNorm(AttNormClass):
    """
    Special Norm class to handle the multiple outputs required by the below Attention class
    """

    def __init__(self, dim, fn):
        super().__init__(dim, fn)

    def forward(self, x, **kwargs):
        x = self.norm(x)

        to_return = self.fn(x, **kwargs)
        if type(to_return) is tuple:
            tmp, *ret = to_return
            return tmp + x, *ret
        else:
            tmp = to_return
            return tmp + x


class AttPostNorm(AttNormClass):
    """
    Special Norm class to handle the multiple outputs required by the below Attention class
    """

    def __init__(self, dim, fn):
        super().__init__(dim, fn)

    def forward(self, x, **kwargs):

        to_return = self.fn(x, **kwargs)
        if type(to_return) is tuple:
            tmp, *ret = to_return
            x = x + tmp

            return self.norm(x), *ret
        else:
            tmp = to_return
            return self.norm(tmp)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_feedforward=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim_feedforward * 2),
            GEGLU(),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, d_model, nhead=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * nhead

        self.scale = dim_head**-0.5
        self.nhead = nhead

        self.to_q = nn.Linear(d_model, inner_dim, bias=False)
        self.to_k = nn.Linear(d_model, inner_dim, bias=False)
        self.to_v = nn.Linear(d_model, inner_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, d_model)

    def forward(self, query, key, value, src_mask=None, return_info=False):

        h = self.nhead

        q = self.to_q(query)  
        k = self.to_k(key)  
        v = self.to_v(value)

        qu, ku, vu = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v)
        )

        sim = einsum("b i d, b j d -> b i j", qu, ku) * self.scale

        if src_mask is not None:
            src_mask = src_mask.bool()
            max_neg_value = -torch.finfo(sim.dtype).max

            src_mask = repeat(src_mask, "b j k -> (b h) j k", h=h)
            sim.masked_fill_(~src_mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, vu)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        if not return_info:
            return self.to_out(out)
        else:
            attn_mean = rearrange(attn, "(b h) m n -> b h m n", h=h).mean(1)
            return self.to_out(out), attn_mean, attn

