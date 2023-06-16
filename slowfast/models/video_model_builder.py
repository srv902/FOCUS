# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""Video models."""

import math
from functools import partial
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from collections import OrderedDict

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.attention import MultiScaleBlock, TrajectoryAttentionBlock
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.stem_helper import PatchEmbed
from slowfast.models.utils import round_width
from .ORViT import ORViT as ORViT

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "2d": [[1, 1, 1]],
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


@MODEL_REGISTRY.register()
class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if cfg.DETECTION.ENABLE:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        1,
                        1,
                    ],
                    [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
                ],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2],
                    ],
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[1][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][2],
                    ],
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
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
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class X3D(nn.Module):
    """
    X3D model builder. It builds a X3D network backbone, which is a ResNet.

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(X3D, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1

        exp_stage = 2.0
        self.dim_c1 = cfg.X3D.DIM_C1

        self.dim_res2 = (
            round_width(self.dim_c1, exp_stage, divisor=8)
            if cfg.X3D.SCALE_RES2
            else self.dim_c1
        )
        self.dim_res3 = round_width(self.dim_res2, exp_stage, divisor=8)
        self.dim_res4 = round_width(self.dim_res3, exp_stage, divisor=8)
        self.dim_res5 = round_width(self.dim_res4, exp_stage, divisor=8)

        self.block_basis = [
            # blocks, c, stride
            [1, self.dim_res2, 2],
            [2, self.dim_res3, 2],
            [5, self.dim_res4, 2],
            [3, self.dim_res5, 2],
        ]
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _round_repeats(self, repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        multiplier = multiplier
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _construct_network(self, cfg):
        """
        Builds a single pathway X3D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        w_mul = cfg.X3D.WIDTH_FACTOR
        d_mul = cfg.X3D.DEPTH_FACTOR
        dim_res1 = round_width(self.dim_c1, w_mul)

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
        )

        # blob_in = s1
        dim_in = dim_res1
        for stage, block in enumerate(self.block_basis):
            dim_out = round_width(block[1], w_mul)
            dim_inner = int(cfg.X3D.BOTTLENECK_FACTOR * dim_out)

            n_rep = self._round_repeats(block[0], d_mul)
            prefix = "s{}".format(
                stage + 2
            )  # start w res2 to follow convention

            s = resnet_helper.ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[block[2]],
                num_blocks=[n_rep],
                num_groups=[dim_inner]
                if cfg.X3D.CHANNELWISE_3x3x3
                else [num_groups],
                num_block_temp_kernel=[n_rep],
                nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
                nonlocal_group=cfg.NONLOCAL.GROUP[0],
                nonlocal_pool=cfg.NONLOCAL.POOL[0],
                instantiation=cfg.NONLOCAL.INSTANTIATION,
                trans_func_name=cfg.RESNET.TRANS_FUNC,
                stride_1x1=cfg.RESNET.STRIDE_1X1,
                norm_module=self.norm_module,
                dilation=cfg.RESNET.SPATIAL_DILATIONS[stage],
                drop_connect_rate=cfg.MODEL.DROPCONNECT_RATE
                * (stage + 2)
                / (len(self.block_basis) + 1),
            )
            dim_in = dim_out
            self.add_module(prefix, s)

        if self.enable_detection:
            NotImplementedError
        else:
            spat_sz = int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE / 32.0))
            self.head = head_helper.X3DHead(
                dim_in=dim_out,
                dim_inner=dim_inner,
                dim_out=cfg.X3D.DIM_C5,
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[cfg.DATA.NUM_FRAMES, spat_sz, spat_sz],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                bn_lin5_on=cfg.X3D.BN_LIN5,
            )

    def forward(self, x, bboxes=None):
        for module in self.children():
            x = module(x)
        return x


@MODEL_REGISTRY.register()
class MViT(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg

        pool_first = cfg.MVIT.POOL_FIRST
        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        use_2d_patch = cfg.MVIT.PATCH_2D
        self.enable_detection = cfg.DETECTION.ENABLE
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        # Prepare output.
        from slowfast.utils import misc
        num_classes = misc.get_num_classes(cfg) #cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.patch_dims[0], embed_dim)
            )
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, embed_dim)
                )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_dim, embed_dim)
            )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][
                1:
            ]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                i
            ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        i_num_frames = self.cfg.DATA.NUM_FRAMES //  (1 if self.cfg.MVIT.PATCH_2D else self.cfg.MVIT.PATCH_STRIDE[0])
        self.blocks = nn.ModuleList()
        self.orvit_blocks = nn.ModuleList()
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(
                embed_dim,
                dim_mul[i + 1],
                divisor=round_width(num_heads, head_mul[i + 1]),
            )
            if i in self.cfg.ORVIT.LAYERS:
                _block = ORViT(
                            cfg=cfg,
                            dim=embed_dim, 
                            dim_out=dim_out,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, 
                            drop=self.drop_rate, 
                            attn_drop=self.drop_rate, 
                            drop_path=dpr[i], 
                            norm_layer=norm_layer, 
                            nb_frames = i_num_frames,
                        )

            else:
                _block = MultiScaleBlock(
                            dim=embed_dim,
                            dim_out=dim_out,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            drop_rate=self.drop_rate,
                            drop_path=dpr[i],
                            norm_layer=norm_layer,
                            kernel_q=pool_q[i] if len(pool_q) > i else [],
                            kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                            stride_q=stride_q[i] if len(stride_q) > i else [],
                            stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                            mode=mode,
                            has_cls_embed=self.cls_embed_on,
                            pool_first=pool_first,
                            ignore_111_kv_kernel=cfg.MVIT.POOL_KV_IGNORE_111_KERNEL,
                        )

            self.blocks.append(_block)

            # Add orvit blocks
            tstride = stride_q[i][0] if (len(stride_q) > i and stride_q[i]) else 1
            if i in self.cfg.ORVIT.ADD_LAYERS:
                assert not (len(stride_q) > i and stride_q[i])
                _block = ORViT(
                            cfg=cfg,
                            dim=embed_dim, 
                            dim_out=dim_out,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, 
                            drop=self.drop_rate, 
                            attn_drop=self.drop_rate, 
                            drop_path=dpr[i], 
                            norm_layer=norm_layer, 
                            nb_frames = i_num_frames,
                        )

            else:
                _block = None
            self.orvit_blocks.append(_block)
            i_num_frames //= tstride

        embed_dim = dim_out
        self.norm = norm_layer(embed_dim)

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[embed_dim],
                num_classes=num_classes,
                pool_size=[[temporal_size // self.patch_stride[0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.TransformerBasicHead(
                embed_dim,
                num_classes,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )
        if self.sep_pos_embed:
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
            if self.cls_embed_on:
                trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.sep_pos_embed:
                if self.cls_embed_on:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                        "cls_token",
                    }
                else:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                    }
            else:
                if self.cls_embed_on:
                    return {"pos_embed", "cls_token"}
                else:
                    return {"pos_embed"}
        else:
            return {}

    def forward(self, x, metadata, bboxes=None):
        x = x[0]
        x = self.patch_embed(x)

        T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        H = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        W = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.patch_dims[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.patch_dims[1] * self.patch_dims[2],
                dim=1,
            )
            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        for blk, blk_orvit in zip(self.blocks, self.orvit_blocks):
            x_prev, thw_prev = x, thw
            x, thw = blk(x_prev,metadata, thw_prev)
            if blk_orvit:
                x_orvit, _ = blk_orvit(x_prev,metadata, thw_prev)
                x = x + x_orvit

        x = self.norm(x)
        if self.enable_detection:
            if self.cls_embed_on:
                x, cls_embed = x[:, 1:], x[:, 0]
            else:
                cls_embed = None
            B, _, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, thw[0], thw[1], thw[2])
            x = self.head([x], bboxes)
        else:
            if self.cls_embed_on:
                x = x[:, 0]
            else:
                x = x.mean(1)
            x_cls = x
            x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class Motionformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, cfg):
        super().__init__()

        self.img_size = cfg.DATA.TRAIN_CROP_SIZE
        self.patch_size = cfg.MF.PATCH_SIZE
        self.in_chans = cfg.MF.CHANNELS
        if cfg.TRAIN.DATASET == "epickitchens":
            self.num_classes = [97, 300]  
        else:
            self.num_classes = cfg.MODEL.NUM_CLASSES
        self.embed_dim = cfg.MF.EMBED_DIM
        self.depth = cfg.MF.DEPTH
        self.num_heads = cfg.MF.NUM_HEADS
        self.mlp_ratio = cfg.MF.MLP_RATIO
        self.qkv_bias = cfg.MF.QKV_BIAS
        self.drop_rate = cfg.MF.DROP
        self.drop_path_rate = cfg.MF.DROP_PATH
        self.head_dropout = cfg.MF.HEAD_DROPOUT
        self.video_input = cfg.MF.VIDEO_INPUT
        self.temporal_resolution = cfg.MF.TEMPORAL_RESOLUTION
        self.use_mlp = cfg.MF.USE_MLP
        self.num_features = self.embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.attn_drop_rate = cfg.MF.ATTN_DROPOUT
        self.head_act = cfg.MF.HEAD_ACT
        self.cfg = cfg

        self.patch_embed_3d = stem_helper.PatchEmbed(
            dim_in=self.in_chans,
            dim_out=self.embed_dim,
            kernel=[self.cfg.MF.PATCH_SIZE_TEMP, self.patch_size, self.patch_size], 
            stride=[self.cfg.MF.PATCH_SIZE_TEMP, self.patch_size, self.patch_size],
            padding=0,
            conv_2d=False,
        )

        self.patch_embed_3d.num_patches = (224 // self.patch_size) ** 2
        self.patch_embed_3d.proj.weight.data = torch.zeros_like(
            self.patch_embed_3d.proj.weight.data)
        # Number of patches
        if self.video_input:
            num_patches = self.patch_embed_3d.num_patches * self.temporal_resolution
        else:
            num_patches = self.patch_embed_3d.num_patches
        self.num_patches = num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed_3d.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)
        trunc_normal_(self.pos_embed, std=.02)

        if self.cfg.MF.POS_EMBED == "joint":
            self.st_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, self.embed_dim))
            trunc_normal_(self.st_embed, std=.02)
        elif self.cfg.MF.POS_EMBED == "separate":
            self.temp_embed = nn.Parameter(
                torch.zeros(1, self.temporal_resolution, self.embed_dim))

        # Layer Blocks
        dpr = [x.item() for x in torch.linspace(
            0, self.drop_path_rate, self.depth)]


        ##
        blocks = []
        for i in range(self.depth):
            _block = None
            if i in self.cfg.ORVIT.LAYERS:
                _block = ORViT(
                            cfg=cfg,
                            dim=self.embed_dim, 
                            num_heads=self.num_heads,
                            mlp_ratio=self.mlp_ratio, 
                            qkv_bias=self.qkv_bias, 
                            drop=self.drop_rate, 
                            attn_drop=self.attn_drop_rate, 
                            norm_layer=norm_layer, 
                            nb_frames = self.temporal_resolution,
                )
            else:
                _block = TrajectoryAttentionBlock(
                    cfg = cfg,
                    dim=self.embed_dim, 
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio, 
                    qkv_bias=self.qkv_bias, 
                    drop=self.drop_rate, 
                    attn_drop=self.attn_drop_rate, 
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )

            blocks.append(_block)
        self.blocks = nn.ModuleList(blocks)

        self.norm = norm_layer(self.embed_dim)

        # MLP head
        if self.use_mlp:
            hidden_dim = self.embed_dim
            if self.head_act == 'tanh':
                print("Using TanH activation in MLP")
                act = nn.Tanh() 
            elif self.head_act == 'gelu':
                print("Using GELU activation in MLP")
                act = nn.GELU()
            else:
                print("Using ReLU activation in MLP")
                act = nn.ReLU()
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, hidden_dim)),
                ('act', act),
            ]))
        else:
            self.pre_logits = nn.Identity()
        
        # Classifier Head
        self.head_drop = nn.Dropout(p=self.head_dropout)
        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            for a, i in enumerate(range(len(self.num_classes))):
                setattr(self, "head%d"%a, nn.Linear(self.embed_dim, self.num_classes[i]))
        else:
            self.head = (nn.Linear(self.embed_dim, self.num_classes) 
                if self.num_classes > 0 else nn.Identity())

        # Initialize weights
        self.init_weights()
        self.apply(self._init_weights)

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.MF.POS_EMBED == "joint":
            return {'pos_embed', 'cls_token', 'st_embed'}
        else:
            return {'pos_embed', 'cls_token', 'temp_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = (nn.Linear(self.embed_dim, num_classes) if num_classes > 0
            else nn.Identity())

    def forward_features(self, x, metadata): # x: [BS, C=3, T=16, H=224, W=224]
        if self.video_input:
            x = x[0]
        B = x.shape[0]
        (B, C,T, H, W) = x.shape

        # Tokenize input
        x = self.patch_embed_3d(x) # [BS, N=T'*H'*W', d]

        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # [BS, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1) # [BS, N, dim]

        # Interpolate positinoal embeddings
        if self.cfg.DATA.TRAIN_CROP_SIZE != 224:
            pos_embed = self.pos_embed
            N = pos_embed.shape[1] - 1
            npatch = int((x.size(1) - 1) / self.temporal_resolution)
            class_emb = pos_embed[:, 0]
            pos_embed = pos_embed[:, 1:]
            dim = x.shape[-1]
            pos_embed = torch.nn.functional.interpolate(
                pos_embed.reshape(
                    1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                    0, 3, 1, 2),
                scale_factor=math.sqrt(npatch / N),
                mode='bicubic',
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            new_pos_embed = torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)
        else:
            new_pos_embed = self.pos_embed
            npatch = self.patch_embed_3d.num_patches

        # Add positional embeddings to input
        if self.video_input:
            if self.cfg.MF.POS_EMBED == "separate":
                cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
                tile_pos_embed = new_pos_embed[:, 1:, :].repeat(
                    1, self.temporal_resolution, 1)
                tile_temporal_embed = self.temp_embed.repeat_interleave(
                    npatch, 1)
                total_pos_embed = tile_pos_embed + tile_temporal_embed
                total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)
                x = x + total_pos_embed
            elif self.cfg.MF.POS_EMBED == "joint":
                x = x + self.st_embed
        else:
            # image input
            x = x + new_pos_embed
                            
        # Apply positional dropout
        x = self.pos_drop(x) # [BS, N, dim]
        # Encoding using transformer layers
        thw = [self.temporal_resolution, int(npatch**0.5), int(npatch**0.5)]
        for i, blk in enumerate(self.blocks):
            x, _ = blk(
                x,
                metadata,
                thw,
            )
        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)
        if not torch.isfinite(x).all():
            print("WARNING: nan in features out")
        return x

    def forward(self, x, metadata): # x: [BS, C=3, T=16, H=224, W=224]
        x = self.forward_features(x, metadata) # [BS, d]
        x = self.head_drop(x)
        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            output = []
            for head in range(len(self.num_classes)):
                x_out = getattr(self, "head%d"%head)(x)
                if not self.training:
                    x_out = torch.nn.functional.softmax(x_out, dim=-1)
                output.append(x_out)
            return output[0], {'verb': output[0], 'noun': output[1]}
        else:
            x = self.head(x)
            if not self.training:
                x = torch.nn.functional.softmax(x, dim=-1)
            return x

# from .STEVE.utils import *
# from .STEVE.dvae import dVAE
# from .STEVE.transformer import TransformerEncoder, TransformerDecoder
# from .build import MODEL_REGISTRY

# class SlotAttentionVideo(nn.Module):
    
#     def __init__(self, num_iterations, num_slots,
#                  input_size, slot_size, mlp_hidden_size,
#                  num_predictor_blocks=1,
#                  num_predictor_heads=4,
#                  dropout=0.1,
#                  epsilon=1e-8):
#         super().__init__()
        
#         self.num_iterations = num_iterations
#         self.num_slots = num_slots
#         self.input_size = input_size
#         self.slot_size = slot_size
#         self.mlp_hidden_size = mlp_hidden_size
#         self.epsilon = epsilon

#         # parameters for Gaussian initialization (shared by all slots).
#         self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
#         self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
#         nn.init.xavier_uniform_(self.slot_mu)
#         nn.init.xavier_uniform_(self.slot_log_sigma)

#         # norms
#         self.norm_inputs = nn.LayerNorm(input_size)
#         self.norm_slots = nn.LayerNorm(slot_size)
#         self.norm_mlp = nn.LayerNorm(slot_size)
        
#         # linear maps for the attention module.
#         self.project_q = linear(slot_size, slot_size, bias=False)
#         self.project_k = linear(input_size, slot_size, bias=False)
#         self.project_v = linear(input_size, slot_size, bias=False)
        
#         # slot update functions.
#         self.gru = gru_cell(slot_size, slot_size)
#         self.mlp = nn.Sequential(
#             linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
#             nn.ReLU(),
#             linear(mlp_hidden_size, slot_size))
#         self.predictor = TransformerEncoder(num_predictor_blocks, slot_size, num_predictor_heads, dropout)

#     def forward(self, inputs):
#         B, T, num_inputs, input_size = inputs.size()

#         # initialize slots
#         slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
#         slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots

#         # setup key and value
#         inputs = self.norm_inputs(inputs)
#         k = self.project_k(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
#         v = self.project_v(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
#         k = (self.slot_size ** (-0.5)) * k
        
#         # loop over frames
#         attns_collect = []
#         slots_collect = []
#         for t in range(T):
#             # corrector iterations
#             for i in range(self.num_iterations):
#                 slots_prev = slots
#                 slots = self.norm_slots(slots)

#                 # Attention.
#                 q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
#                 attn_logits = torch.bmm(k[:, t], q.transpose(-1, -2))
#                 attn_vis = F.softmax(attn_logits, dim=-1)
#                 # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

#                 # Weighted mean.
#                 attn = attn_vis + self.epsilon
#                 attn = attn / torch.sum(attn, dim=-2, keepdim=True)
#                 updates = torch.bmm(attn.transpose(-1, -2), v[:, t])
#                 # `updates` has shape: [batch_size, num_slots, slot_size].

#                 # Slot update.
#                 slots = self.gru(updates.view(-1, self.slot_size),
#                                  slots_prev.view(-1, self.slot_size))
#                 slots = slots.view(-1, self.num_slots, self.slot_size)

#                 # use MLP only when more than one iterations
#                 if i < self.num_iterations - 1:
#                     slots = slots + self.mlp(self.norm_mlp(slots))

#             # collect
#             attns_collect += [attn_vis]
#             slots_collect += [slots]

#             # predictor
#             slots = self.predictor(slots)

#         attns_collect = torch.stack(attns_collect, dim=1)   # B, T, num_inputs, num_slots
#         slots_collect = torch.stack(slots_collect, dim=1)   # B, T, num_slots, slot_size

#         return slots_collect, attns_collect


# class LearnedPositionalEmbedding1D(nn.Module):

#     def __init__(self, num_inputs, input_size, dropout=0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.pe = nn.Parameter(torch.zeros(1, num_inputs, input_size), requires_grad=True)
#         nn.init.trunc_normal_(self.pe)

#     def forward(self, input, offset=0):
#         """
#         input: batch_size x seq_len x d_model
#         return: batch_size x seq_len x d_model
#         """
#         T = input.shape[1]
#         return self.dropout(input + self.pe[:, offset:offset + T])


# class CartesianPositionalEmbedding(nn.Module):

#     def __init__(self, channels, image_size):
#         super().__init__()

#         self.projection = conv2d(4, channels, 1)
#         self.pe = nn.Parameter(self.build_grid(image_size).unsqueeze(0), requires_grad=False)

#     def build_grid(self, side_length):
#         coords = torch.linspace(0., 1., side_length + 1)
#         coords = 0.5 * (coords[:-1] + coords[1:])
#         grid_y, grid_x = torch.meshgrid(coords, coords)
#         return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

#     def forward(self, inputs):
#         # `inputs` has shape: [batch_size, out_channels, height, width].
#         # `grid` has shape: [batch_size, in_channels, height, width].
#         # print("size of inputs >> ", inputs.shape)
#         # print("size of the proj pe >> ", self.projection(self.pe).shape)
#         # exit()
#         return inputs + self.projection(self.pe)

# class OneHotDictionary(nn.Module):
#     def __init__(self, vocab_size, emb_size):
#         super().__init__()
#         self.dictionary = nn.Embedding(vocab_size, emb_size)

#     def forward(self, x):
#         """
#         x: B, N, vocab_size
#         """

#         tokens = torch.argmax(x, dim=-1)  # batch_size x N
#         token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
#         return token_embs


# class STEVEEncoder(nn.Module):
#     def __init__(self, args):
#         super().__init__()
        
#         self.cnn = nn.Sequential(
#             Conv2dBlock(args.model.img_channels, args.model.cnn_hidden_size, 5, 1 if args.model.image_size == 64 else 2, 2),
#             Conv2dBlock(args.model.cnn_hidden_size, args.model.cnn_hidden_size, 5, 1, 2),
#             Conv2dBlock(args.model.cnn_hidden_size, args.model.cnn_hidden_size, 5, 1, 2),
#             conv2d(args.model.cnn_hidden_size, args.decoder.d_model, 5, 1, 2),
#         )

#         self.pos = CartesianPositionalEmbedding(args.decoder.d_model, args.model.image_size if args.model.image_size == 64 else args.model.image_size // 2)

#         self.layer_norm = nn.LayerNorm(args.decoder.d_model)

#         self.mlp = nn.Sequential(
#             linear(args.decoder.d_model, args.decoder.d_model, weight_init='kaiming'),
#             nn.ReLU(),
#             linear(args.decoder.d_model, args.decoder.d_model))

#         self.savi = SlotAttentionVideo(
#             args.model.num_iterations, args.model.num_slots,
#             args.decoder.d_model, args.model.slot_size, args.model.mlp_hidden_size,
#             args.model.num_predictor_blocks, args.model.num_predictor_heads, args.model.predictor_dropout)

#         self.slot_proj = linear(args.model.slot_size, args.decoder.d_model, bias=False)


# class STEVEDecoder(nn.Module):
#     def __init__(self, args):
#         super().__init__()

#         self.dict = OneHotDictionary(args.decoder.vocab_size, args.decoder.d_model)

#         self.bos = nn.Parameter(torch.Tensor(1, 1, args.decoder.d_model))
#         nn.init.xavier_uniform_(self.bos)

#         self.pos = LearnedPositionalEmbedding1D(1 + (args.model.image_size // 4) ** 2, args.decoder.d_model)

#         self.tf = TransformerDecoder(
#             args.decoder.num_blocks, (args.model.image_size // 4) ** 2, args.decoder.d_model, args.decoder.num_heads, args.decoder.dropout)

#         self.head = linear(args.decoder.d_model, args.decoder.vocab_size, bias=False)

# @MODEL_REGISTRY.register()
# class STEVE(nn.Module):
#     """
#     STEVE module working on video clips with slots to capture object related features
#     """
#     def __init__(self, args):
#         super().__init__()
        
#         self.num_iterations = args.model.num_iterations
#         self.num_slots = args.model.num_slots
#         self.cnn_hidden_size = args.model.cnn_hidden_size
#         self.slot_size = args.model.slot_size
#         self.mlp_hidden_size = args.model.mlp_hidden_size
#         self.img_channels = args.model.img_channels
#         self.image_size = args.model.image_size
#         self.vocab_size = args.decoder.vocab_size
#         self.d_model = args.decoder.d_model

#         # dvae
#         self.dvae = dVAE(args.decoder.vocab_size, args.model.img_channels)

#         # encoder networks
#         self.steve_encoder = STEVEEncoder(args)

#         # decoder networks
#         self.steve_decoder = STEVEDecoder(args)

#     def forward(self, video, tau, hard):
#         B, T, C, H, W = video.size()

#         video_flat = video.flatten(end_dim=1)                               # B * T, C, H, W

#         # dvae encode
#         z_logits = F.log_softmax(self.dvae.encoder(video_flat), dim=1)       # B * T, vocab_size, H_enc, W_enc
#         z_soft = gumbel_softmax(z_logits, tau, hard, dim=1)                  # B * T, vocab_size, H_enc, W_enc
#         z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()         # B * T, vocab_size, H_enc, W_enc
#         z_hard = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                         # B * T, H_enc * W_enc, vocab_size
#         z_emb = self.steve_decoder.dict(z_hard)                                                     # B * T, H_enc * W_enc, d_model
#         z_emb = torch.cat([self.steve_decoder.bos.expand(B * T, -1, -1), z_emb], dim=1)             # B * T, 1 + H_enc * W_enc, d_model
#         z_emb = self.steve_decoder.pos(z_emb)                                                       # B * T, 1 + H_enc * W_enc, d_model

#         # dvae recon
#         dvae_recon = self.dvae.decoder(z_soft).reshape(B, T, C, H, W)               # B, T, C, H, W
#         dvae_mse = ((video - dvae_recon) ** 2).sum() / (B * T)                      # 1

#         # savi
#         # print("video flat >> ", video_flat.shape)
#         emb = self.steve_encoder.cnn(video_flat)      # B * T, cnn_hidden_size, H, W
#         # print("size of emb >> ", emb.shape)
#         emb = self.steve_encoder.pos(emb)             # B * T, cnn_hidden_size, H, W
#         H_enc, W_enc = emb.shape[-2:]

#         emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                                   # B * T, H * W, cnn_hidden_size
#         emb_set = self.steve_encoder.mlp(self.steve_encoder.layer_norm(emb_set))                            # B * T, H * W, cnn_hidden_size
#         emb_set = emb_set.reshape(B, T, H_enc * W_enc, self.d_model)                                                # B, T, H * W, cnn_hidden_size

#         slots, attns = self.steve_encoder.savi(emb_set)         # slots: B, T, num_slots, slot_size
#                                                                 # attns: B, T, num_slots, num_inputs

#         attns = attns\
#             .transpose(-1, -2)\
#             .reshape(B, T, self.num_slots, 1, H_enc, W_enc)\
#             .repeat_interleave(H // H_enc, dim=-2)\
#             .repeat_interleave(W // W_enc, dim=-1)          # B, T, num_slots, 1, H, W
#         attns = video.unsqueeze(2) * attns + (1. - attns)                               # B, T, num_slots, C, H, W

#         # decode
#         slots = self.steve_encoder.slot_proj(slots)                                                         # B, T, num_slots, d_model
#         pred = self.steve_decoder.tf(z_emb[:, :-1], slots.flatten(end_dim=1))                               # B * T, H_enc * W_enc, d_model
#         pred = self.steve_decoder.head(pred)                                                                # B * T, H_enc * W_enc, vocab_size
#         cross_entropy = -(z_hard * torch.log_softmax(pred, dim=-1)).sum() / (B * T)                         # 1

#         return (dvae_recon.clamp(0., 1.),
#                 cross_entropy,
#                 dvae_mse,
#                 attns)

#     def encode(self, video):
#         B, T, C, H, W = video.size()

#         video_flat = video.flatten(end_dim=1)

#         # savi
#         emb = self.steve_encoder.cnn(video_flat)      # B * T, cnn_hidden_size, H, W
#         emb = self.steve_encoder.pos(emb)             # B * T, cnn_hidden_size, H, W
#         H_enc, W_enc = emb.shape[-2:]

#         emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                                   # B * T, H * W, cnn_hidden_size
#         emb_set = self.steve_encoder.mlp(self.steve_encoder.layer_norm(emb_set))                            # B * T, H * W, cnn_hidden_size
#         emb_set = emb_set.reshape(B, T, H_enc * W_enc, self.d_model)                                                # B, T, H * W, cnn_hidden_size

#         slots, attns = self.steve_encoder.savi(emb_set)     # slots: B, T, num_slots, slot_size
#                                                             # attns: B, T, num_slots, num_inputs

#         attns = attns \
#             .transpose(-1, -2) \
#             .reshape(B, T, self.num_slots, 1, H_enc, W_enc) \
#             .repeat_interleave(H // H_enc, dim=-2) \
#             .repeat_interleave(W // W_enc, dim=-1)                      # B, T, num_slots, 1, H, W

#         attns_vis = video.unsqueeze(2) * attns + (1. - attns)           # B, T, num_slots, C, H, W

#         return slots, attns_vis, attns

#     def decode(self, slots):
#         B, num_slots, slot_size = slots.size()
#         H_enc, W_enc = (self.image_size // 4), (self.image_size // 4)
#         gen_len = H_enc * W_enc

#         slots = self.steve_encoder.slot_proj(slots)

#         # generate image tokens auto-regressively
#         z_gen = slots.new_zeros(0)
#         input = self.steve_decoder.bos.expand(B, 1, -1)
#         for t in range(gen_len):
#             decoder_output = self.steve_decoder.tf(
#                 self.steve_decoder.pos(input),
#                 slots
#             )
#             z_next = F.one_hot(self.steve_decoder.head(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size)
#             z_gen = torch.cat((z_gen, z_next), dim=1)
#             input = torch.cat((input, self.steve_decoder.dict(z_next)), dim=1)

#         z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, H_enc, W_enc)
#         gen_transformer = self.dvae.decoder(z_gen)

#         return gen_transformer.clamp(0., 1.)

#     def reconstruct_autoregressive(self, video):
#         """
#         image: batch_size x img_channels x H x W
#         """
#         B, T, C, H, W = video.size()
#         slots, attns, _ = self.encode(video)
#         recon_transformer = self.decode(slots.flatten(end_dim=1))
#         recon_transformer = recon_transformer.reshape(B, T, C, H, W)

#         return recon_transformer
