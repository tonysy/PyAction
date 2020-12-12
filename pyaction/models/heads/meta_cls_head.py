#!/usr/bin/env python3
# Copyright (c) ShanghaiTech PLUS Group. All Rights Reserved.

import torch
import torch.nn as nn
import math

from torch.nn import functional as F


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

        self.meta_cls_layer = cfg.build_meta_cls_head(cfg)
        self.sem_cls_layer = cfg.build_sem_cls_head(cfg)

        # for few-shot
        self.n_support_way = cfg.META.SETTINGS.N_SUPPORT_WAY
        self.k_support_shot = cfg.META.SETTINGS.K_SUPPORT_SHOT
        self.n_query_way = cfg.META.SETTINGS.N_QUERY_WAY
        self.k_query_shot = cfg.META.SETTINGS.K_QUERY_SHOT
        self.meta_loss_scale = cfg.META.LOSS.META_SCALE
        self.sem_loss_scale = cfg.META.LOSS.SEM_SCALE

    def forward(self, inputs,
        support_meta_labels,
        query_meta_labels=None,
        all_sem_labels=None,
    ):
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

        # for semantic branch
        if self.sem_cls_layer:
            if self.training:
                all_sem_cls_scores = self.sem_cls_layer(x.mean(-1))
            else:
                all_sem_cls_scores = None
        else:
            all_sem_cls_scores = None

        # for meta learning branch
        support_feat, query_feat = torch.split(
            x.view(-1, S, C, mT),
            [
                self.n_support_way * self.k_support_shot,
                self.n_query_way * self.k_query_shot,
            ],
            dim=1,
        )

        # (B,K_query_shot, N_way)
        query_meta_cls_scores = self.meta_cls_layer(
            support_feat,
            query_feat,
            support_meta_labels
        ).view(-1, self.n_support_way)


        if self.training:
            loss = MetaOutputs(
                meta_pred=query_meta_cls_scores,
                sem_pred=all_sem_cls_scores,
                meta_gt=query_meta_labels,
                sem_gt=all_sem_labels,
                meta_loss_scale=self.meta_loss_scale,
                sem_loss_scale=self.sem_loss_scale,
            ).losses()
            return query_meta_cls_scores, all_sem_cls_scores, loss
        else:
            return query_meta_cls_scores


class MetaOutputs(object):
    """
    A class that stores information about outputs of a Meta Network Head.
    """

    def __init__(self,
        meta_pred, sem_pred=None,
        meta_gt=None, sem_gt=None,
        meta_loss_scale=1.0, sem_loss_scale=1.0,
    ):
        """
        Args:
            meta_pred (Tensor):
                box2box transform instance for proposal-to-detection transformations.
            sem_pred (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            meta_gt (Tensor): 
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            sem_gt (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            meta_loss_scale (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            sem_loss_scale (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.meta_pred = meta_pred
        self.sem_pred = None if sem_pred is None else sem_pred

        self.meta_gt = None if meta_gt is None else meta_gt
        self.sem_gt = None if sem_gt is None else sem_gt

        self.meta_loss_scale = meta_loss_scale
        self.sem_loss_scale = sem_loss_scale

    def meta_ce_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        return F.cross_entropy(self.meta_pred, self.meta_gt, reduction="mean")

    def sem_ce_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        if self.sem_pred is not None:
            return F.cross_entropy(self.sem_pred, self.sem_gt, reduction="mean")
        else:
            return 0

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_sem": self.sem_loss_scale * self.sem_ce_loss(),
            "loss_meta": self.meta_loss_scale * self.meta_ce_loss(),
        }
