#!/usr/bin/env python3
# Copyright (c) ShanghaiTech PLUS Group. All Rights Reserved.

"""Video models."""
import torch
import torch.nn as nn

from pyaction.models import head_helper, stem_helper
# from pyaction.models.video_model_builder import ResNetModel

# from pyaction.models.video_model_builder import _MODEL_STAGE_DEPTH
# from pyaction.models.video_model_builder import _TEMPORAL_KERNEL_BASIS
# from pyaction.models.video_model_builder import _POOL1
from .shadow_resnet import _POOL1
from .shadow_resnet import _MODEL_STAGE_DEPTH
from .shadow_resnet import _TEMPORAL_KERNEL_BASIS

from . import resnet_helper
from .shadow_resnet import ShadowResNetModel

class MyResNetModel(ShadowResNetModel):
    """
    Revised: Add support for LatentGNN Module 
    Author: Songyang Zhang
    --------------------------------
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, SlowOnly).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "Slowfast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """
    def __init__(self, cfg):
        super(MyResNetModel, self).__init__(cfg)

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model with support of LatentGNN

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            # dim_out=[width_per_group * 4],
            dim_out=[width_per_group],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            lgnn_inds=cfg.LATENTGNN3D.LOCATION[0],
            lgnn_group=cfg.LATENTGNN3D.GROUP[0],
            num_nodes=cfg.LATENTGNN3D.NUM_NODES[0],
            instantiation=cfg.LATENTGNN3D.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            **cfg.LATENTGNN3D.CONFIG
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 2],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            lgnn_inds=cfg.LATENTGNN3D.LOCATION[1],
            lgnn_group=cfg.LATENTGNN3D.GROUP[1],
            num_nodes=cfg.LATENTGNN3D.NUM_NODES[1],
            instantiation=cfg.LATENTGNN3D.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            **cfg.LATENTGNN3D.CONFIG
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 2],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            lgnn_inds=cfg.LATENTGNN3D.LOCATION[2],
            lgnn_group=cfg.LATENTGNN3D.GROUP[2],
            num_nodes=cfg.LATENTGNN3D.NUM_NODES[2],
            instantiation=cfg.LATENTGNN3D.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            **cfg.LATENTGNN3D.CONFIG
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            lgnn_inds=cfg.LATENTGNN3D.LOCATION[3],
            lgnn_group=cfg.LATENTGNN3D.GROUP[3],
            num_nodes=cfg.LATENTGNN3D.NUM_NODES[3],
            instantiation=cfg.LATENTGNN3D.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            **cfg.LATENTGNN3D.CONFIG
        )

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 8],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func="sigmoid",
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 8],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
            )
