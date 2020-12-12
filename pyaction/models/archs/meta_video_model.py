#!/usr/bin/env python3
# Copyright (c) ShanghaiTech PLUS Group. All Rights Reserved.

import torch

from torch import nn
import pyaction.utils.weight_init_helper as init_helper


class MetaVideoModel(nn.Module):
    """
    ResNet model builder for meta learning. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, SlowOnly).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "Slowfast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """
    def __init__(self, cfg):
        super(MetaVideoModel, self).__init__()

        # self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = cfg.build_backbone(cfg)
        self.head = cfg.build_head(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )
        self.enable_detection = cfg.DETECTION.ENABLE

    def forward(self,x,
        support_meta_label,
        query_meta_labels=None,
        all_sem_labels=None,
        bboxes=None):
        x = torch.cat(x, dim=1)
        B, S, _, T, H, W = x.shape
        x = [x.view(B * S, -1, T, H, W)]

        # [B*S, C, T, H, W]
        x = self.backbone(x)

        if self.enable_detection:
            # x = self.head(x, bboxes)
            raise NotImplementedError
        else:
            return self.head(
                    x,
                    support_meta_label,
                    query_meta_labels,
                    all_sem_labels
                )