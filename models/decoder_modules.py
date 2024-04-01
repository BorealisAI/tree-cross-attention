# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from torch import nn

##### DecoderLayer


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()

    def forward(self, layer_outputs):
        raise NotImplementedError


class CADecoderLayer(DecoderLayer):
    def __init__(self):
        super(CADecoderLayer, self).__init__()

    def forward(self, query_encodings, context_memory):
        query_pred_encodings = context_memory.retrieve(query_encodings)
        return query_pred_encodings


class TCADecoderLayer(DecoderLayer):
    # Tree Cross Attention Decoder Layer
    def __init__(self):
        super(TCADecoderLayer, self).__init__()

    def forward(self, query_encodings, context_memory):
        if self.training:
            (
                x_tca,
                x_tca_leaf,
                entropy_scores,
                log_action_probs,
            ) = context_memory.retrieve(query_encodings)
            return x_tca, x_tca_leaf, entropy_scores, log_action_probs
        else:
            x_tca = context_memory.retrieve(query_encodings)
            return x_tca


##### Decoder


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, query_encoding, context_memory):
        """
        Input:
            - query_encodings: [B, M, D] Tensor
            - context_memory: Memory
        Returns:
            - query_pred_encodings: [B, M, D] Tensor
        """
        raise NotImplementedError


class CADecoder(Decoder):
    def __init__(self, decoder_layer):
        super(CADecoder, self).__init__()
        self.layer = decoder_layer

    def forward(self, query_encoding, context_memory):
        x = query_encoding
        x = self.layer(x, context_memory)
        return x


class TCADecoder(Decoder):
    # Tree Cross Attention Decoder
    def __init__(self, decoder_layer):
        super(TCADecoder, self).__init__()
        self.layer = decoder_layer

    def forward(self, query_encodings, context_memory):
        x = query_encodings
        if self.training:
            x_tca, x_tca_leaf, entropy_scores, log_action_probs = self.layer(x, context_memory)
            return x_tca, x_tca_leaf, entropy_scores, log_action_probs
        else:
            x = self.layer(x, context_memory)
            return x
