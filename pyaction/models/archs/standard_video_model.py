#!/usr/bin/env python3
# Copyright (c) ShanghaiTech PLUS Group. All Rights Reserved.

from torch import nn
import pyaction.utils.weight_init_helper as init_helper


class StandardVideoModel(nn.Module):
    def __init__(self, cfg):
        super(StandardVideoModel, self).__init__()

        # self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = cfg.build_backbone(cfg)
        self.head = cfg.build_head(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )
        self.enable_detection = cfg.DETECTION.ENABLE
        # self.to(self.device)

    def forward(self, x, bboxes=None):
        x = self.backbone(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)

        return x
