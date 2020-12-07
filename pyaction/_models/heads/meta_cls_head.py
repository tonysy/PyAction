#!/usr/bin/env python3
# Copyright (c) ShanghaiTech PLUS Group. All Rights Reserved.

import torch
import torch.nn as nn
import math

# from torch.nn import functional as F


class MetaBasicHead(nn.Module):
    """
    Classification head for few-shot video classification

    """

    def __init__(self, cfg, pool_size):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
        """
        super(MetaBasicHead, self).__init__()
        # the list of channel dimensions of the p inputs to the ResNetHead.
        dim_in = [cfg.RESNET.WIDTH_PER_GROUP * 32]
        # dropout rate. If equal to 0.0, perform no dropout.
        dropout_rate = cfg.MODEL.DROPOUT_RATE
        # activation function to use.
        act_func = cfg.MODEL.HEAD_ACT
        # the channel dimensions of the p outputs to the ResNetHead.
        # num_classes = cfg.MODEL.NUM_CLASSES

        pool_size = [
            [
                # cfg.DATA.NUM_FRAMES // pool_size[0][0],
                1,
                int(math.ceil(cfg.DATA.CROP_SIZE / 32) // pool_size[0][1]),
                int(math.ceil(cfg.DATA.CROP_SIZE / 32) // pool_size[0][2]),
            ]
        ]  # None for AdaptiveAvgPool3d((1, 1, 1))

        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."

        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            # avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

        self.metric_layer = cfg.build_meta_metric(cfg)

        # for few-shot
        self.n_support_way = cfg.META.SETTINGS.N_SUPPORT_WAY
        self.k_support_shot = cfg.META.SETTINGS.K_SUPPORT_SHOT
        self.n_query_way = cfg.META.SETTINGS.N_QUERY_WAY
        self.k_query_shot = cfg.META.SETTINGS.K_QUERY_SHOT

    def forward(self, inputs, support_labels):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)

        pool_out = []
        # to handle the two pathway cases(e.g. slowfast)
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)

        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.squeeze(-1).squeeze(-1)
        _, C, mT = x.shape
        S = (
            self.n_support_way * self.k_support_shot
            + self.n_query_way * self.k_query_shot
        )

        support_feat, query_feat = torch.split(
            x.view(-1, S, C, mT),
            [
                self.n_support_way * self.k_support_shot,
                self.n_query_way * self.k_query_shot,
            ],
            dim=1,
        )

        # (B,K_query_shot, N_way)
        scores = self.metric_layer(support_feat, query_feat, support_labels).view(
            -1, self.n_support_way
        )

        return scores
        # _, C, mT, mH, mW = x.shape
        # S = self.n_support_way * k_support_shot + \
        #         self.n_query_way * self.k_query_shot

        # x = torch.split(
        #     x.view(-1, S, C, mT, mH, mW),
        #     [self.n_support_way * k_support_shot,
        #      self.n_query_way * self.k_query_shot],
        #     dim=1)
