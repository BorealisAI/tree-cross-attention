# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F
from attrdict import AttrDict

SOS = 0  # <SOS> value
EOS = 1  # <EOS> value


class RandomMirroredSampler(object):
    def __init__(self, sequence_length, seed=None, num_chars=256):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.sequence_length = sequence_length
        self.num_chars = num_chars

    def sample(self, batch_size=16, device="cpu"):
        batch = AttrDict()

        # Randomly generate tokens (skipping the SOS and EOS token values)
        rand_seq = torch.randint(
            low=2,
            high=self.num_chars + 2,
            size=(batch_size, self.sequence_length // 2 - 1),
        )

        yt_seq = torch.cat(
            (torch.flip(rand_seq, dims=[-1]), torch.ones((batch_size, 1)) * EOS), dim=1
        )

        sos_token = torch.ones((batch_size, 1)) * SOS
        xc_seq = torch.cat((sos_token, rand_seq), dim=1)

        onehot_xc = (
            F.one_hot(xc_seq.long(), num_classes=2 + self.num_chars).float().to(device)
        )
        onehot_yt = (
            F.one_hot(yt_seq.long(), num_classes=2 + self.num_chars).float().to(device)
        )

        batch.xc = onehot_xc  
        batch.xt = torch.zeros(batch.xc.shape).to(device)
        batch.x = torch.cat((batch.xc, batch.xt), 1) 
        batch.yt = onehot_yt  

        return batch
