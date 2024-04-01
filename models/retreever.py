# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn

from models.modules import build_mlp

from torch import nn
import torch.nn.functional as F
from attrdict import AttrDict

from torch.distributions.normal import Normal

from models.encoder_modules import Encoder, TransformerEncoderLayer, NullEncoderLayer
from models.decoder_modules import (
    CADecoderLayer,
    CADecoder,
    TCADecoderLayer,
    TCADecoder,
)
from models.processor_modules import VanillaProcessor, TreeProcessor
from models.positional_encoding import PositionalEncoding


class Retreever(nn.Module):
    def __init__(
        self,
        dim_x,
        dim_y,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        norm_first=True,  
        encoder_type="quadratic",  # quadratic or constant computation encoder
        decoder_type="tca",  # (tca) or (ca)
        bptt=False,  # Backpropagate through tree
        branch_factor=2,  # how many children per node
        ca_loss_weight=1.0,
        entropy_bonus_weight=0.01,
        rl_loss_weight=1.0,
        num_aggregation_layers=1,
        predictor_type="classification",
        loss="nll",
        is_metalearning=False,
        is_sequential_data=False,
        aggregator_type="transformer",
        classification_rew_type="acc",
        dim_xc=None,  # dim_x of contexts
        dim_xt=None,  # dim_x of targets
        bound_std=False,
        heuristic="none",
    ):
        super(Retreever, self).__init__()
        self.decoder_type = decoder_type
        self.ca_loss_weight = ca_loss_weight
        self.is_metalearning = is_metalearning
        self.is_sequential_data = is_sequential_data
        self.branch_factor = branch_factor
        self.entropy_bonus_weight = entropy_bonus_weight
        self.rl_loss_weight = rl_loss_weight
        self.classification_rew_type = classification_rew_type
        self.bound_std = bound_std
        self.heuristic = heuristic
        self.loss = loss  # nll or mse

        if dim_xc is None:
            dim_xc = dim_x
        if dim_xt is None:
            dim_xt = dim_x

        if self.heuristic == "random_proj":
            generator = torch.Generator()
            generator.manual_seed(2147483647)
            rand_mat = torch.randn((dim_xc, 1), generator=generator)
            self.rand_proj_mat = nn.parameter.Parameter(rand_mat, requires_grad=False)

        # Context Related:
        if self.is_metalearning:
            self.embedder = build_mlp(
                dim_xc + dim_y, d_model, d_model, emb_depth, activation_type="ELU"
            )
        else:
            self.embedder = build_mlp(
                dim_xc, d_model, d_model, emb_depth, activation_type="ELU"
            )

        # Query Related
        self.query_embedder = build_mlp(
            dim_xt, d_model, d_model, emb_depth, activation_type="ELU"
        )

        # Positional Encoding
        if self.is_sequential_data:
            self.embedder = nn.Sequential(
                self.embedder,
                PositionalEncoding(d_model, dropout),
                build_mlp(
                    d_model, d_model * 2, d_model, depth=3, activation_type="ReLU"
                ),
            )
            self.query_embedder = nn.Sequential(
                self.query_embedder,
                PositionalEncoding(d_model, dropout),
                build_mlp(
                    d_model, d_model * 2, d_model, depth=3, activation_type="ReLU"
                ),
            )

        # Encoders
        if encoder_type == "quadratic":
            encoder_layer = TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, norm_first=norm_first
            )
            self.encoder = Encoder(encoder_layer, num_layers)
        elif encoder_type == "constant":
            encoder_layer = NullEncoderLayer()
            self.encoder = Encoder(encoder_layer, num_layers)
        else:
            raise NotImplementedError


        # Decoders
        if decoder_type == "tca":  # Generates a tree structure of the data
            self.processor = TreeProcessor(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                norm_first,
                branch_factor,
                num_aggregation_layers,
                bptt,
                aggregator_type,
            )
        elif decoder_type == "ca":
            self.processor = VanillaProcessor(
                d_model, nhead, dim_feedforward, dropout, norm_first
            )
        else:
            raise NotImplementedError

        if decoder_type == "tca":
            decoder_layer = TCADecoderLayer()
            self.decoder = TCADecoder(decoder_layer)
        elif decoder_type == "ca":
            decoder_layer = CADecoderLayer()
            self.decoder = CADecoder(decoder_layer)
        else:
            raise NotImplementedError

        # Predictors
        self.norm_first = norm_first
        if self.norm_first:
            self.norm = nn.LayerNorm(d_model)
        if predictor_type == "classification" or predictor_type == "regression":
            self.predictor = build_mlp(
                d_model, d_model * 2, dim_y, depth=2, activation_type="ReLU"
            )
        elif predictor_type == "uncertainty_regression":
            self.predictor = build_mlp(
                d_model, d_model * 2, dim_y * 2, depth=2, activation_type="ReLU"
            )
        else:
            raise NotImplementedError

        self.predictor_type = predictor_type

    def reset(self):
        # Resets the tree for future generation
        self.processor.reset()

    def process_data(self, xc, yc):
        # Process data organizes the data for easy k-d tree construction.
        # In our case, we consider splitting according to a single axis, making the data organization simple (equivalent to sorting)
        # Organizes the data for tree generation

        if self.heuristic == "none":
            # GP Regression and Copy Task data are already ordered for easy construction
            pass
        elif self.heuristic == "sort_x1":
            orig_xc_shape = xc.shape
            orig_yc_shape = yc.shape
            xc_values = xc[:, :, 0]  # select the first index

            sorted_ftx_idxs = torch.argsort(xc_values).reshape(-1)
            batch_idxs = (
                torch.arange(xc.shape[0], device="cuda")
                .repeat_interleave(xc.shape[1])
                .reshape(-1)
            )

            xc = xc[batch_idxs, sorted_ftx_idxs, :].reshape(orig_xc_shape)
            yc = yc[batch_idxs, sorted_ftx_idxs, :].reshape(orig_yc_shape)

        elif self.heuristic == "sort_x2":
            orig_xc_shape = xc.shape
            orig_yc_shape = yc.shape
            xc_values = xc[:, :, 1]  # select the second index

            sorted_ftx_idxs = torch.argsort(xc_values).reshape(-1)
            batch_idxs = (
                torch.arange(xc.shape[0], device="cuda")
                .repeat_interleave(xc.shape[1])
                .reshape(-1)
            )

            xc = xc[batch_idxs, sorted_ftx_idxs, :].reshape(orig_xc_shape)
            yc = yc[batch_idxs, sorted_ftx_idxs, :].reshape(orig_yc_shape)
        elif self.heuristic == "random_proj":
            orig_xc_shape = xc.shape
            orig_yc_shape = yc.shape
            xc_data_proj = xc @ self.rand_proj_mat

            sorted_ftx_idxs = torch.argsort(xc_data_proj.squeeze(-1)).reshape(-1)
            batch_idxs = (
                torch.arange(xc.shape[0], device="cuda")
                .repeat_interleave(xc.shape[1])
                .reshape(-1)
            )

            xc = xc[batch_idxs, sorted_ftx_idxs, :].reshape(orig_xc_shape)
            yc = yc[batch_idxs, sorted_ftx_idxs, :].reshape(orig_yc_shape)
        else:
            raise NotImplementedError
        return xc, yc

    def process_context(self, xc, yc=None):
        assert (self.is_metalearning and yc is not None) or (
            not self.is_metalearning and yc is None
        )

        if self.heuristic != "none":
            xc, yc = self.process_data(xc, yc)

        if self.is_metalearning:
            x_y_ctx = torch.cat((xc, yc), dim=-1)
            context_embeddings = self.embedder(x_y_ctx)
        else:
            context_embeddings = self.embedder(xc)

        context_encoding = self.encoder(context_embeddings)
        context_memory_block = self.processor(context_encoding)

        return context_memory_block

    def predict_dist(self, encoding):
        out = self.predictor(encoding)
        assert self.predictor_type == "uncertainty_regression"

        mean, std = torch.chunk(out, 2, dim=-1)
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)
        out = Normal(mean, std)

        return out

    def predict(self, batch):  # Makes the prediction
        if self.is_metalearning:
            context_memory_block = self.process_context(batch.xc, batch.yc)
        else:
            context_memory_block = self.process_context(batch.xc)

        query_embedding = self.query_embedder(batch.xt)

        encoding = self.decoder(query_embedding, context_memory_block)
        if self.predictor_type == "uncertainty_regression":
            return self.predict_dist(encoding)
        else:
            return self.predictor(encoding)

    def make_prediction(self, encoding):
        if self.predictor_type == "uncertainty_regression":
            return self.predict_dist(encoding)
        else:
            return self.predictor(encoding)

    def forward(self, batch):  # Computes Training Loss
        if self.is_metalearning:
            context_memory_block = self.process_context(batch.xc, batch.yc)
        else:
            context_memory_block = self.process_context(batch.xc)

        query_embedding = self.query_embedder(batch.xt)

        if self.decoder_type == "tca":
            if self.training:
                tca_encoding, tca_leaf_encoding, entropy, log_action_probs = self.decoder(query_embedding, context_memory_block)
            else:
                tca_encoding = self.decoder(query_embedding, context_memory_block)
        else:
            tca_encoding = self.decoder(query_embedding, context_memory_block)

        tca_prediction = self.make_prediction(tca_encoding)
        if self.training and self.decoder_type == "tca":
            tca_leaf_prediction = self.make_prediction(tca_leaf_encoding)

        outs = AttrDict()
        outs = self.tca_loss(outs, tca_prediction, batch.yt)

        if self.training and self.decoder_type == "tca":
            outs = self.tca_leaf_loss(outs, tca_leaf_prediction, batch.yt)
            outs = self.rl_loss(
                outs, tca_prediction, batch.yt, log_action_probs, entropy
            )
            outs.loss = (
                outs.tca_loss
                + self.ca_loss_weight * outs.tca_leaf_loss
                + self.rl_loss_weight * outs.rl_loss
            )
        else:
            outs.loss = outs.tca_loss

        return outs

    def tca_loss(self, outs, prediction, yt):  # $\mathcal{L}_{TCA}$ loss term
        if self.loss == "nll":
            tar_ll = prediction.log_prob(yt).sum(-1)
            outs.tar_ll = tar_ll.mean()
            loss_value = -outs.tar_ll
        elif self.loss == "mse": 
            loss = nn.MSELoss(reduction="none")
            outs.mse = loss(prediction, yt).mean()
            loss_value = outs.mse
        elif self.loss == "ce":
            loss = nn.CrossEntropyLoss(reduction="none")
            outs.ce = loss(
                prediction.flatten(0, -2), yt.max(-1)[-1].flatten(0, -1)
            ).mean()
            outs.acc = (
                (prediction.flatten(0, -2).max(-1)[1] == yt.max(-1)[-1].flatten(0, -1))
                .float()
                .mean()
            )
            loss_value = outs.ce
        else:
            raise NotImplementedError

        if self.decoder_type == "tca":
            outs.tca_loss = loss_value
        else:
            outs.loss = loss_value
        return outs

    def tca_leaf_loss(
        self, outs, tca_leaf_prediction, level_yt
    ):  # $\mathcal{L}_{CA}$ loss term
        if self.loss == "nll":
            outs.tca_leaf_loss = -tca_leaf_prediction.log_prob(level_yt).sum(-1).mean()
        elif self.loss == "mse": 
            loss = nn.MSELoss(reduction="none")
            outs.tca_leaf_loss = loss(tca_leaf_prediction, level_yt).mean()
        elif self.loss == "ce":
            loss = nn.CrossEntropyLoss(reduction="none")
            outs.tca_leaf_loss = loss(
                tca_leaf_prediction.flatten(0, -2), level_yt.max(-1)[-1].flatten(0, -1)
            ).mean()
        else:
            raise NotImplementedError

        return outs

    def rl_loss(
        self, outs, prediction, yt, log_action_probs, entropy
    ):  # $\mathcal{L}_{RL}$ loss term
        if self.loss == "nll":
            tca_tar_ll = prediction.log_prob(yt).sum(-1)
            rl_rew = tca_tar_ll.detach().flatten()
            baseline_value = 0.0
        elif self.loss == "mse": 
            loss = nn.MSELoss(reduction="none")
            tca_mse = loss(prediction, yt)
            rl_rew = -tca_mse.detach()
            baseline_value = 0.0
        elif self.loss == "ce":
            if self.classification_rew_type == "nce":
                loss = nn.CrossEntropyLoss(reduction="none")
                tca_ce = loss(prediction.flatten(0, -2), yt.max(-1)[-1].flatten(0, -1))
                rl_rew = -tca_ce.detach()
                baseline_value = -torch.log(
                    torch.tensor(prediction.shape[-1])
                )  # -log(# classes)
            elif self.classification_rew_type == "acc":
                _, pred_idxes = prediction.max(-1)
                _, label_idxes = yt.max(-1)
                rl_rew = (pred_idxes == label_idxes).float().detach().flatten()
                baseline_value = torch.tensor(0)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        outs.entropy_bonus = entropy
        outs.log_action_probs = log_action_probs.mean()
        outs.rl_loss = (
            -(log_action_probs * (rl_rew - baseline_value)).mean()
            - self.entropy_bonus_weight * entropy
        )  # REINFORCE + Entropy regularization
        return outs
