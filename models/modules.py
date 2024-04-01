# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch.nn as nn


def build_mlp(dim_in, dim_hid, dim_out, depth, activation_type):
    if activation_type == "ELU":
        activation = nn.ELU
    elif activation_type == "ReLU":
        activation = nn.ReLU
    else:
        raise NotImplementedError
    modules = [nn.Linear(dim_in, dim_hid), activation(inplace=True)]
    for _ in range(depth - 2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(activation(inplace=True))
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)
