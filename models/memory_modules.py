# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from models.attention_modules import *
import math
from models.aggregator_modules import TransformerAggregator

##### Memory


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def retrieve(self, x):
        """
        Arguments:
            x: [B, M, D] Tensor
        Returns:
            ret: [B, M, D] Tensor
        """
        raise NotImplementedError


class VanillaMemory(Memory):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm_first):
        super(VanillaMemory, self).__init__()

        if norm_first:
            Norm = PreNorm
        else:
            Norm = PostNorm

        self.cross_attention = Norm(
            d_model,
            Attention(d_model, nhead=nhead, dim_head=d_model // nhead, dropout=dropout),
        )
        self.cross_attention_ff = Norm(
            d_model, FeedForward(d_model, dim_feedforward, dropout)
        )

    def setup_data(self, layer_data):
        self.layer_data = layer_data

    def reset(self):
        self.layer_data = None

    def retrieve(self, query_data):
        """
        Arguments:
            query_data: [B, M, D] Tensor
        Returns:
            ret: [B, M, D] Tensor
        """
        ca_output = self.cross_attention(
            query_data, key=self.layer_data, value=self.layer_data
        )
        ca_ff_output = self.cross_attention_ff(ca_output)
        return ca_ff_output


class TreeMemory(Memory):
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
        super(TreeMemory, self).__init__()
        self.norm_first = norm_first
        if norm_first:
            Norm = PreNorm
            AttNorm = AttPreNorm
        else:
            Norm = PostNorm
            AttNorm = AttPostNorm

        # Check Notes for implementation
        self.branch_factor = branch_factor

        self.train_tree_data = ( # List representation of the tree
            [] # To be filled with type: (layer_data, layer_mask)
        )  

        if aggregator_type == "transformer":
            # Used to order the datapoints.
            self.aggregator = TransformerAggregator(
                num_aggregation_layers,
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                norm_first,
                bptt,
            )
        else:
            raise NotImplementedError

        # The attention model and policy
        self.query_model = AttNorm(
            d_model,
            Attention(d_model, nhead=nhead, dim_head=d_model // nhead, dropout=dropout),
        )
        self.query_ff = Norm(
            d_model,
            FeedForward(d_model, dim_feedforward=dim_feedforward, dropout=dropout),
        )


    def setup_data(self, layer_data):
        self.tree_generator(layer_data)

    def reset(self):
        del self.train_tree_data
        self.train_tree_data = []

    def pad_node_data(self, node_data):
        # Pad node data to make it easily handleable

        device = node_data.device
        B, N, D = node_data.shape

        k = (
            torch.floor(
                torch.log(torch.tensor(N - 1))
                / torch.log(torch.tensor(self.branch_factor))
            )
            .int()
            .item()
        )
        P = math.ceil(N / self.branch_factor**k)

        num_pad_nodes = (
            P * (self.branch_factor**k) - N
        )

        tree_data = torch.zeros(B, N + num_pad_nodes, D).to(device) 
        tree_data[:, :N, :] = node_data
        tree_data[:, N:, :] = (
            node_data[:, :num_pad_nodes, :] + 1e-5
        )  # Pad with real data and some buffer to relatively evenly split the data
        mask = torch.zeros(tree_data.shape[:-1], device=device).unsqueeze(
            -1
        )  # [B, N', 1]
        mask[:, :N, :] = 1  # Real data
        return tree_data, mask

    def tree_generator(self, node_data):
        # For efficiency, we avoid an explicit k-d tree construction and use a tensor representation of a tree structure.
        # This tree construction leverages the fact that the node_data was previously organized. 
        # An example of a tensor representation of the leaves is [2][2][2][2] (i.e., a 2x2x2x2 tensor)
        # Example 1: Indexing according to [0][1][1][0] refers to the leaf achieved by going left, right, right, left down the tree.
        # Example 2: Indexing according to [1][1][0][0] refers to the leaf achieved by going right, right, left, left down the tree.

        B, N, D = node_data.shape
        k = (
            torch.floor(
                torch.log(torch.tensor(N - 1))
                / torch.log(torch.tensor(self.branch_factor))
            )
            .int()
            .item()
        )  
        P = math.ceil(N / self.branch_factor**k)

        tree_depth_data, tree_depth_mask = self.pad_node_data(
            node_data
        )  # Pad the data to an easily organizable size

        tree_depth_data = tree_depth_data.reshape(
            B, *([self.branch_factor] * k), P, D
        )  # Construct the leaves
        tree_depth_mask = tree_depth_mask.reshape(B, *([self.branch_factor] * k), P, 1)

        self.bottom_up_aggregation(tree_depth_data, tree_depth_mask)

    def bottom_up_aggregation(self, tree_depth_data, tree_depth_mask):
        D = tree_depth_data.shape[-1]
        # tree_depth_data: [B, b, b, ..., b, P, D], tree_depth_mask: [B, b, b, ..., b, P, 1]
        self.train_tree_data = [(tree_depth_data, tree_depth_mask.detach())]

        # Perform the bottom-up aggregation
        while len(tree_depth_data.shape) > 2:  # Stops at [B, D]
            tmp_batch_size = np.prod(tree_depth_data.shape[:-2])
            branch_size = tree_depth_data.shape[-2]

            tmp_tree_depth_data = tree_depth_data.reshape(
                (tmp_batch_size, branch_size, D)
            )
            tmp_tree_depth_mask = tree_depth_mask.reshape(
                (tmp_batch_size, branch_size, 1)
            )

            computed_tree_depth_data = self.aggregator(
                tmp_tree_depth_data, tmp_tree_depth_mask
            ).squeeze(
                -2
            )  # [B..., D]

            computed_tree_depth_mask = tmp_tree_depth_mask.any(
                dim=-2
            ).float()  # if any of its children are not padding nodes, then it is not a padding node

            tree_depth_data = computed_tree_depth_data.reshape(
                (*(tree_depth_data.shape[:-2]), tree_depth_data.shape[-1])
            )  # [B, b, ..., b, D]
            tree_depth_mask = computed_tree_depth_mask.reshape(
                (*(tree_depth_mask.shape[:-2]), tree_depth_mask.shape[-1])
            )  # [B, b, ..., b, 1]

            self.train_tree_data.append((tree_depth_data, tree_depth_mask.detach()))

        self.train_tree_data = list(reversed(self.train_tree_data))

    def retrieve(self, query_data):
        entropy_att_scores_list = []
        log_branch_sel_prob_list = []

        pred_emb, entropy_att_scores_list, log_branch_sel_prob_list = (
            self.tree_retrieval(query_data)
        )

        if self.norm_first:
            ret_emb = self.query_ff(pred_emb + query_data)
        else:
            ret_emb = self.query_ff(pred_emb)

        if not self.training:
            return ret_emb
        else:
            leaf_pred_emb = self.tree_leaves_retrieval(query_data)
            entropy_scores, log_action_probs = self.process_rl_terms(
                entropy_att_scores_list, log_branch_sel_prob_list
            )

            if self.norm_first:  # Standard PreNorm used in CA but this time for TCA
                leaf_ret_emb = self.query_ff(leaf_pred_emb + query_data)
            else:
                leaf_ret_emb = self.query_ff(leaf_pred_emb)

            return ret_emb, leaf_ret_emb, entropy_scores, log_action_probs

    def tree_retrieval(self, query_data):
        device = query_data.device
        batch_size, nQueries = query_data.shape[0], query_data.shape[1]
        B = batch_size
        M = nQueries
        D = self.train_tree_data[0][0].shape[-1]

        filtered_tree_data = self.train_tree_data[1:]
        flattened_query_data = query_data.flatten(0, 1).unsqueeze(1)  # [B*M, 1, D]

        entropy_att_scores_list = []
        log_branch_sel_prob_list = []
        selected_data_embeddings = [] # Array of [B*M, 1, D] (Stores selected nodes)
        selected_data_masks = [] # Array of [B*M, 1, 1] (Stores selected nodes' mask)
        
        for i in range(len(filtered_tree_data)):
            (layer_data_embeddings, layer_data_mask) = filtered_tree_data[i]
            if i == 0:
                # Root Node
                layer_data_embeddings = repeat(layer_data_embeddings, 'B b d -> B M b d', M = M)
                layer_data_embeddings = rearrange(layer_data_embeddings, 'B M b d -> (B M) b d')
                # [B*M, N_0, D]

                layer_data_mask = repeat(layer_data_mask, 'B b d -> B M b d', M = M)
                layer_data_mask = rearrange(layer_data_mask, 'B M b d -> (B M) b d')
                # [B*M, N_0, 1]

                N_0 = layer_data_embeddings.shape[1]
                layer_data_indices = repeat(torch.arange(N_0, device=device), 'n -> b n', b = B*M)
                # [B*M, N_0]
            else:
                # Leaf or intermediate Nodes
                # layer_data_embeddings: [B, b, b, b, D]
                layer_data_embeddings = rearrange(layer_data_embeddings, 'ba ... b D -> ba (...) b D')
                layer_data_mask = rearrange(layer_data_mask, 'ba ... b D -> ba (...) b D')

                next_branch_size = layer_data_embeddings.shape[-2]

                repeated_idxes = repeat(torch.arange(B, device=device), 'n -> n k', k = M).flatten()
                # [B*M]: [0,1,...,B, 0,1,...,B, ]

                # next_layer_data_indices: [B*M, 1]
                layer_data_embeddings = layer_data_embeddings[
                    repeated_idxes, next_layer_data_indices.flatten()
                ].reshape(B * M, 1, next_branch_size, D)
                layer_data_mask = layer_data_mask[
                    repeated_idxes, next_layer_data_indices.flatten()
                ].reshape(B * M, 1, next_branch_size, 1)

                layer_data_embeddings = rearrange(layer_data_embeddings, 'B a c D -> B (a c) D')
                # [B, b, D]
                layer_data_mask = rearrange(layer_data_mask, 'B a c D -> B (a c) D')
                # [B, b, 1]

                if i == len(filtered_tree_data) - 1:
                    selected_data_embeddings.append(layer_data_embeddings)
                    selected_data_masks.append(layer_data_mask)
                    break


                # Track indices of selected nodes
                tmp_indices = rearrange(torch.arange(next_branch_size, device=device), 'b -> 1 1 b') # [1, 1, b]

                layer_data_indices = repeat(next_layer_data_indices * next_branch_size, 'B M -> B M b', b = next_branch_size) + tmp_indices
                layer_data_indices = rearrange(layer_data_indices, 'B M b -> (B M) b')
                # layer_data_indices: [B*M, b]

            N_i = layer_data_embeddings.shape[1]  # b 

            _, level_search_att_weight_mean_nodes, search_att_weight = self.query_model(
                flattened_query_data,
                key=layer_data_embeddings,
                value=layer_data_embeddings,
                src_mask=rearrange(layer_data_mask, 'BM b 1 -> BM 1 b'),
                return_info=True,
            )# [B*M, 1, b]
                
            # Select the next node to expand
            if self.training:
                # Stochastic selection
                selected_indices = torch.multinomial(level_search_att_weight_mean_nodes.flatten(0, 1), 1)
            else:  
                # Greedily (deterministically) select the nodes to expand
                selected_indices = level_search_att_weight_mean_nodes.flatten(0, 1).max(-1)[1].unsqueeze(-1)

            # Compute the mask for the selected/rejected nodes 
            tree_search_level_embeddings = layer_data_embeddings.reshape(B*M, N_i, D)
            tree_search_level_mask = (1 - F.one_hot(selected_indices, num_classes = N_i)).reshape(B*M, N_i, 1)

            # Add the level's node embeddings and mask
            selected_data_embeddings.append(tree_search_level_embeddings) # Add to "S"
            selected_data_masks.append(tree_search_level_mask)

            # Compute next tree layer indices
            next_layer_data_indices = layer_data_indices[
                torch.arange(B * M, device="cuda"), selected_indices.flatten()
            ].reshape(
                B * M, 1
            )

            # Compute additional terms for training
            if self.training:
                # Compute Entropy Bonus Entropy Bonus
                entropy_att_scores_list.append(
                    (-search_att_weight * torch.log(search_att_weight + 1e-9)).sum(-1)
                ) 
                # Compute action log probabilities
                log_branch_sel_prob = torch.log(level_search_att_weight_mean_nodes.squeeze(1)[
                        torch.arange(B * M, device="cuda"), selected_indices.flatten()
                    ].squeeze(-1))
                log_branch_sel_prob_list.append(log_branch_sel_prob)

        # Aggregate the selected nodes
        search_data_embeddings = torch.cat(selected_data_embeddings, dim=1)
        search_data_masks = torch.cat(selected_data_masks, dim=1).transpose(1, 2)
        
        # Using the aggregated nodes, compute the final embedding
        # pred_emb_pre_out: [B*M, 1, D], flattened_query_Data:[B*M, 1, D], search_data_embeddings: [B*M, N_i, D], search_masks: [B*M, 1, N_i]
        pred_emb = self.query_model(
            flattened_query_data,
            key=search_data_embeddings,
            value=search_data_embeddings,
            src_mask=search_data_masks,
            return_info=False,
        )

        # Reshape the embedding to the correct representation for Attention
        pred_emb = pred_emb.reshape(B, M, D)

        return pred_emb, entropy_att_scores_list, log_branch_sel_prob_list

    def tree_leaves_retrieval(self, query_data):
        M = query_data.shape[1]
        # For computing L_{TCA}
        leaf_data_embeddings, leaf_data_mask = self.train_tree_data[-1]
        leaf_data_embeddings = leaf_data_embeddings.flatten(1, -2)
        leaf_data_mask = leaf_data_mask.flatten(1, -2).transpose(1, 2).repeat(1, M, 1)
        leaf_pred_emb = self.query_model(
            query_data,
            key=leaf_data_embeddings,
            value=leaf_data_embeddings,
            src_mask=leaf_data_mask,
            return_info=False,
        )
        return leaf_pred_emb

    def process_rl_terms(self, entropy_att_scores_list, log_branch_sel_prob_list):
        # For computing L_{RL}
        if len(entropy_att_scores_list) > 0:
            entropy_scores = torch.stack(entropy_att_scores_list, dim=2)
            entropy_scores = entropy_scores.mean()
        else:
            entropy_scores = torch.tensor(0.0).cuda()

        if len(log_branch_sel_prob_list) > 0:
            log_action_probs = torch.stack(log_branch_sel_prob_list, dim=-1)
            log_action_probs = log_action_probs.sum(-1)
        else:
            log_action_probs = torch.tensor(0.0).cuda()
        return entropy_scores, log_action_probs
