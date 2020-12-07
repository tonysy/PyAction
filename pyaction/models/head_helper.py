#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn

from pyaction.layers import ROIAlign
import math

class ResNetRoIHead(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d([pool_size[pathway][0], 1, 1], stride=1)
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

    def forward(self, inputs, bboxes):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(inputs[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, bboxes)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # B C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection(x)
        x = self.act(x)
        return x


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self, dim_in, num_classes, pool_size, dropout_rate=0.0, act_func="softmax",\
            get_feature=False, feature_dim=None, debug=False, no_spatial_pool=False):
        self.debug = debug
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.

            Few-shot:
            get_feature: whether to return feature vector
            feature_dim: length of the feature vector
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        self.get_feature = get_feature
        self.feature_dim = feature_dim

        self.no_spatial_pool = no_spatial_pool  ###

        for pathway in range(self.num_pathways):
            avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.

        if not self.get_feature:
            self.projection = nn.Linear(sum(dim_in), 1000, bias=True)  #### num_classes

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

    def forward(self, inputs):

        if self.debug:
            import pdb; pdb.set_trace()

        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []

        # for pathway in range(self.num_pathways):
        #     m = getattr(self, "pathway{}_avgpool".format(pathway))
        #     pool_out.append(m(inputs[pathway]))

        if self.no_spatial_pool:
            for pathway in range(self.num_pathways):
                pool_out.append(inputs[pathway])
        else:
            for pathway in range(self.num_pathways):
                m = getattr(self, "pathway{}_avgpool".format(pathway))
                pool_out.append(m(inputs[pathway]))

        x = torch.cat(pool_out, 1)

        # assert x.shape[-1] == x.shape[-2] == 1

        if self.debug:
            import pdb; pdb.set_trace()

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        # import pdb; pdb.set_trace()
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        # if self.debug:
        #     import pdb; pdb.set_trace()

        if not self.get_feature:
            x = self.projection(x)

            # Commented for g2
            # # Performs fully convlutional inference.
            # if not self.training:
            #     x = self.act(x)
            #     x = x.mean([1, 2, 3])

        # if self.debug:
        #     import pdb; pdb.set_trace()

        # x = x.view(x.shape[0], -1)
        # when test:
        # RuntimeError: view size is not compatible with input tensor's size and stride 
        # (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        x = x.reshape(x.shape[0], -1)

        # print(x.shape, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # if self.debug:
        #     import pdb; pdb.set_trace()

        return x


class MetaClsHead(nn.Module):
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
        super(MetaClsHead, self).__init__()
        # the list of channel dimensions of the p inputs to the ResNetHead.
        dim_in = [cfg.RESNET.WIDTH_PER_GROUP * 32]
        # dropout rate. If equal to 0.0, perform no dropout.
        dropout_rate = cfg.MODEL.DROPOUT_RATE
        # activation function to use.
        act_func = 'softmax' # cfg.MODEL.HEAD_ACT
        # the channel dimensions of the p outputs to the ResNetHead.
        # num_classes = cfg.MODEL.NUM_CLASSES

        # if cfg.MULTIGRID.SHORT_CYCLE:
        #     pool_size = [None, None]
        # else:
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
            # if pool_size[pathway] is None:
            #     avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            # else:
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
        from .meta_metric import CosSimMetric
        # self.metric_layer = cfg.build_meta_metric(cfg)
        self.metric_layer = CosSimMetric(cfg)
        # for few-shot
        self.n_support_way = cfg.META.N_SUPPORT_WAY
        self.k_support_shot = cfg.META.K_SUPPORT_SHOT
        self.n_query_way = cfg.META.N_QUERY_WAY
        self.k_query_shot = cfg.META.K_QUERY_SHOT


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
        
        # import pdb; pdb.set_trace()
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
