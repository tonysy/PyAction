#!/usr/bin/env python3
# Copyright (c) ShanghaiTech PLUS Group. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F


def convert_one_hot(labels, N_way):
    """
    Args:
        labels: (B, N_way*K_shot)
    """
    B, NK, _ = labels.shape
    # import pdb; pdb.set_trace()
    onehot_labels = torch.zeros(B, NK, N_way).to(
        labels.device
    )  # the last dim as one-hot
    onehot_labels.scatter_(2, labels, 1.0)

    return onehot_labels


class CosSimMetric(nn.Module):
    """Cosine Similarity Head for video clip"""

    def __init__(self, cfg):
        super(CosSimMetric, self).__init__()

        self.n_support_way = cfg.META.N_SUPPORT_WAY
        self.k_support_shot = cfg.META.K_SUPPORT_SHOT
        if cfg.META.COSINE_SCALE_LEARN:
            self.scale = nn.Parameter(torch.ones(1) * cfg.META.COSINE_SCALE)
        else:
            self.scale = cfg.META.COSINE_SCALE

    def forward(self, support, query, support_labels):
        """
        Args:
            supports (torch.Tensor): Shape:(B, N_way*K_shot, C, T)
            query (torch.Tensor): Shape:(B, K_query_shot, C, T)
        """

        normed_support = F.normalize(support, dim=-2)
        normed_query = F.normalize(query, dim=-2)

        B, _, C, T = normed_support.shape
        # normed_support = normed_support.view(
        #     B, self.n_support_way, self.k_support_shot, C, T
        # )

        # (B, N_way*K_shot, C)
        support_vector = torch.mean(normed_support, dim=-1)
        # (B, K_query_shot, C)
        query_vector = torch.mean(normed_query, dim=-1)
        # (B, K_query_shot, C) * (B, C, N_way*K_shot) -> (B,K_query_shot, N_way*K_shot)
        scores = torch.einsum("abc,adc->abd", query_vector, support_vector)
        onehot_labels = convert_one_hot(support_labels, self.n_support_way)
        # (B, K_query_shot,  N_way*K_shot) * (B,  N_way*K_shot, N_way) ->(B, K_query_shot, N_way)
        scores = torch.bmm(scores, onehot_labels)
        scores = self.scale * scores
        # print(self.scale)
        return scores
