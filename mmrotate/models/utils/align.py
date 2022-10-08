# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmcv.ops import (DeformConv2d, DeformConv2dPack, ModulatedDeformConv2d,
                      rotated_feature_align)
from mmengine.model import BaseModule, Sequential, normal_init
from torch import Tensor, nn

from mmrotate.registry import TASK_UTILS


@TASK_UTILS.register_module()
class AlignConv(BaseModule):
    """AlignConv."""

    def __init__(self, feat_channels, kernel_size, strides, deform_groups=1):
        super(AlignConv, self).__init__()
        self.feat_channels = feat_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.deform_conv = DeformConv2d(
            self.feat_channels,
            self.feat_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.deform_conv, std=0.01)

    @torch.no_grad()
    def get_offset(self, anchors, featmap_size, stride):
        """Get the offset of AlignConv."""
        dtype, device = anchors.dtype, anchors.device
        feat_h, feat_w = featmap_size
        pad = (self.kernel_size - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # get sampling locations of default conv
        xc = torch.arange(0, feat_w, device=device, dtype=dtype)
        yc = torch.arange(0, feat_h, device=device, dtype=dtype)
        yc, xc = torch.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        x_conv = xc[:, None] + xx
        y_conv = yc[:, None] + yy

        # get sampling locations of anchors
        x_ctr, y_ctr, w, h, a = torch.unbind(anchors, dim=1)
        x_ctr, y_ctr, w, h = \
            x_ctr / stride, y_ctr / stride, \
            w / stride, h / stride
        cos, sin = torch.cos(a), torch.sin(a)
        dw, dh = w / self.kernel_size, h / self.kernel_size
        x, y = dw[:, None] * xx, dh[:, None] * yy
        xr = cos[:, None] * x - sin[:, None] * y
        yr = sin[:, None] * x + cos[:, None] * y
        x_anchor, y_anchor = xr + x_ctr[:, None], yr + y_ctr[:, None]
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = torch.stack([offset_y, offset_x], dim=-1)
        offset = offset.reshape(anchors.size(0),
                                -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        return offset

    def forward_single(self, x, anchor, stride):
        """Forward function for single level."""
        num_imgs, _, H, W = x.shape
        offset_list = [
            self.get_offset(anchor[i].reshape(-1, 5), (H, W), stride)
            for i in range(num_imgs)
        ]
        offset_tensor = torch.stack(offset_list, dim=0)
        x = self.deform_conv(x, offset_tensor)
        x = self.relu(x)
        return x

    def forward(self, x: List[Tensor],
                anchors: List[List[Tensor]]) -> List[Tensor]:
        """Forward function."""
        mlvl_anchors = []
        for i in range(len(x)):
            anchor = torch.stack([anchor[i] for anchor in anchors], dim=0)
            mlvl_anchors.append(anchor)
        out = []
        for x, anchor, stride in zip(x, mlvl_anchors, self.strides):
            out.append(self.forward_single(x, anchor, stride))
        return out


@TASK_UTILS.register_module()
class RepAlignConv(AlignConv):
    """AlignConv."""

    def __init__(self,
                 feat_channels,
                 kernel_size,
                 strides,
                 deploy=False,
                 deform_groups=1):
        super(AlignConv, self).__init__()
        self.feat_channels = feat_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.deploy = deploy
        self.deform_groups = deform_groups
        self.norm_cfg = dict(type='BN')
        assert kernel_size == 3

        if self.deploy:
            self.deform_conv_reparam = DeformConv2d(
                self.feat_channels,
                self.feat_channels,
                kernel_size=self.kernel_size,
                padding=(self.kernel_size - 1) // 2,
                deform_groups=self.deform_groups)
        else:
            self.deform_conv_3x3 = DeformConv2d(
                self.feat_channels,
                self.feat_channels,
                kernel_size=3,
                padding=1,
                deform_groups=self.deform_groups)
            self.deform_conv_1x1 = DeformConv2d(
                self.feat_channels,
                self.feat_channels,
                kernel_size=1,
                padding=0,
                deform_groups=self.deform_groups)
            self.identity = DeformConv2d(
                self.feat_channels,
                self.feat_channels,
                kernel_size=3,
                padding=1,
                deform_groups=self.deform_groups)

        self.relu = build_activation_layer(dict(type='ReLU'))

    def init_weights(self):
        """Initialize weights of the head."""
        if self.deploy:
            normal_init(self.deform_conv_reparam, std=0.01)
        else:
            normal_init(self.deform_conv_3x3, std=0.01)
            normal_init(self.deform_conv_1x1, std=0.01)
            identity_weight = torch.zeros(
                (self.feat_channels, self.feat_channels, 3, 3),
                dtype=self.identity.weight.dtype)
            for i in range(self.feat_channels):
                identity_weight[i, i % self.feat_channels, 1, 1] = 1
            identity_weight = identity_weight.to(self.identity.weight.device)
            self.identity.weight.data = identity_weight

    def train(self, mode=True):
        super().train(mode)
        if not self.deploy:
            self.identity.eval()
            for param in self.identity.parameters():
                param.requires_grad = False

    def forward_single(self, x, anchor, stride):
        """Forward function for single level."""
        num_imgs, _, H, W = x.shape
        offset_list = [
            self.get_offset(anchor[i].reshape(-1, 5), (H, W), stride)
            for i in range(num_imgs)
        ]
        offset_tensor_3x3 = torch.stack(offset_list, dim=0)

        if self.deploy:
            out = self.deform_conv_reparam(x, offset_tensor_3x3)
        else:
            offset_tensor_1x1 = offset_tensor_3x3[:, 8:10, :, :].contiguous()
            out = self.deform_conv_3x3(x, offset_tensor_3x3) + \
                  self.deform_conv_1x1(x, offset_tensor_1x1) + \
                  self.identity(x, offset_tensor_3x3)

        out = self.relu(out)
        return out

    def forward(self, x: List[Tensor],
                anchors: List[List[Tensor]]) -> List[Tensor]:
        """Forward function."""
        mlvl_anchors = []
        for i in range(len(x)):
            anchor = torch.stack([anchor[i] for anchor in anchors], dim=0)
            mlvl_anchors.append(anchor)
        out = []
        for x, anchor, stride in zip(x, mlvl_anchors, self.strides):
            out.append(self.forward_single(x, anchor, stride))
        return out

    def reparameterize(self):
        """Fuse all the parameters of all branches.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Parameters after fusion of all
                branches. the first element is the weights and the second is
                the bias.
        """
        weight_3x3 = self._get_conv_weight(self.deform_conv_3x3)
        weight_1x1 = self._get_conv_weight(self.deform_conv_1x1)
        weight_id = self._get_conv_weight(self.identity)
        # pad a conv1x1 weight to a conv3x3 weight
        weight_1x1 = F.pad(weight_1x1, [1, 1, 1, 1], value=0)

        return weight_3x3 + weight_1x1 + weight_id

    def _get_conv_weight(self, conv):
        if conv is None:
            return 0
        return conv.weight

    def switch_to_deploy(self):
        """Switch the model structure from training mode to deployment mode."""
        if self.deploy:
            return

        reparam_weight = self.reparameterize()
        self.deform_conv_reparam = DeformConv2d(
            self.feat_channels,
            self.feat_channels,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
            deform_groups=self.deform_groups)
        self.deform_conv_reparam.weight.data = reparam_weight

        for param in self.parameters():
            param.detach_()
        delattr(self, 'deform_conv_3x3')
        delattr(self, 'deform_conv_1x1')
        delattr(self, 'identity')

        self.deploy = True


# class MDCNBN(BaseModule):
#
#     def __init__(self,
#                  feat_channels,
#                  kernel_size,
#                  padding=0,
#                  deform_groups=1,
#                  norm_cfg=dict(type='BN')):
#         super(MDCNBN, self).__init__()
#         self.conv = ModulatedDeformConv2d(
#             feat_channels,
#             feat_channels,
#             kernel_size=kernel_size,
#             padding=padding,
#             deform_groups=deform_groups,
#             bias=False
#         )
#         # self.norm = build_norm_layer(norm_cfg,
#         #                              num_features=feat_channels)[1]
#
#     def init_weights(self):
#         normal_init(self.conv, std=0.01)
#
#     def forward(self, x, offset, mask):
#         x = self.conv(x, offset, mask)
#         # x = self.norm(x)
#         return x


@TASK_UTILS.register_module()
class WeightedRepAlignConv(AlignConv):
    """AlignConv."""

    def __init__(self,
                 feat_channels,
                 kernel_size,
                 strides,
                 deploy=False,
                 deform_groups=1):
        super(AlignConv, self).__init__()
        self.feat_channels = feat_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.deploy = deploy
        self.deform_groups = deform_groups
        self.norm_cfg = dict(type='BN')
        assert kernel_size == 3

        self.factor = nn.Embedding(5, 2)
        if self.deploy:
            self.deform_conv_reparam = ModulatedDeformConv2d(
                self.feat_channels,
                self.feat_channels,
                kernel_size=self.kernel_size,
                padding=(self.kernel_size - 1) // 2,
                deform_groups=self.deform_groups,
                bias=False)
        else:
            self.deform_conv_3x3 = ModulatedDeformConv2d(
                feat_channels,
                feat_channels,
                kernel_size=3,
                padding=1,
                deform_groups=deform_groups,
                bias=False)
            self.deform_conv_1x1 = ModulatedDeformConv2d(
                feat_channels,
                feat_channels,
                kernel_size=1,
                padding=0,
                deform_groups=deform_groups,
                bias=False)

        self.relu = build_activation_layer(dict(type='ReLU'))

    def init_weights(self):
        """Initialize weights of the head."""
        nn.init.constant_(self.factor.weight, 0.5)
        if self.deploy:
            normal_init(self.deform_conv_reparam, std=0.01)
        else:
            normal_init(self.deform_conv_3x3, std=0.01)
            normal_init(self.deform_conv_1x1, std=0.01)

    def forward_single(self, feats, anchor, score, stride, factor):
        """Forward function for single level."""
        num_imgs, _, H, W = feats[0].shape
        offset_list = [
            self.get_offset(anchor[i].reshape(-1, 5), (H, W), stride)
            for i in range(num_imgs)
        ]
        offset_tensor_3x3 = torch.stack(offset_list, dim=0)
        score = score.detach().max(dim=1, keepdim=True)[0]

        out_feats = tuple()
        for fn, x in enumerate(feats):
            _factor = factor[fn]

            score_1x1 = score.sigmoid() * _factor + (1 - _factor)
            score_3x3 = score_1x1.repeat(1, 9, 1, 1)

            if self.deploy:
                out = self.deform_conv_reparam(x, offset_tensor_3x3, score_3x3)
            else:
                offset_tensor_1x1 = offset_tensor_3x3[:,
                                                      8:10, :, :].contiguous()
                out = self.deform_conv_3x3(x, offset_tensor_3x3, score_3x3) + \
                      self.deform_conv_1x1(x, offset_tensor_1x1, score_1x1)
            out = self.relu(out)
            out_feats = out_feats + (out, )
        return out_feats

    def forward(self, x: List[Tuple[Tensor]], anchors: List[List[Tensor]],
                scores: List[Tensor]) -> List[Tensor]:
        """Forward function."""
        level = len(x)
        mlvl_anchors = []
        for i in range(level):
            anchor = torch.stack([anchor[i] for anchor in anchors], dim=0)
            mlvl_anchors.append(anchor)

        factors = self.factor.weight.clone().clamp(min=0, max=1)
        # factors = self.factor.weight.clone().sigmoid()

        out = []
        for i, (feats, anchor, score, stride) in enumerate(
                zip(x, mlvl_anchors, scores, self.strides)):
            factor = factors[i]
            out.append(
                self.forward_single(feats, anchor, score, stride, factor))

        return out

    def reparameterize(self):
        """Fuse all the parameters of all branches.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Parameters after fusion of all
                branches. the first element is the weights and the second is
                the bias.
        """
        weight_3x3 = self._get_conv_weight(self.deform_conv_3x3)
        weight_1x1 = self._get_conv_weight(self.deform_conv_1x1)
        # pad a conv1x1 weight to a conv3x3 weight
        weight_1x1 = F.pad(weight_1x1, [1, 1, 1, 1], value=0)

        return weight_3x3 + weight_1x1

    def _get_conv_weight(self, conv):
        if conv is None:
            return 0
        return conv.weight

    def switch_to_deploy(self):
        """Switch the model structure from training mode to deployment mode."""
        if self.deploy:
            return

        reparam_weight = self.reparameterize()
        self.deform_conv_reparam = ModulatedDeformConv2d(
            self.feat_channels,
            self.feat_channels,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
            deform_groups=self.deform_groups,
            bias=False)
        self.deform_conv_reparam.weight.data = reparam_weight

        for param in self.parameters():
            param.detach_()
        delattr(self, 'deform_conv_3x3')
        delattr(self, 'deform_conv_1x1')
        # delattr(self, 'branch_norm')

        self.deploy = True


@TASK_UTILS.register_module()
class WeightedAlignConv(AlignConv):

    def __init__(self, feat_channels, kernel_size, strides, deform_groups=1):
        super(AlignConv, self).__init__()
        self.feat_channels = feat_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.deform_groups = deform_groups
        self.norm_cfg = dict(type='BN')
        assert kernel_size == 3

        self.factor = nn.Embedding(5, 2)
        self.deform_conv = ModulatedDeformConv2d(
            self.feat_channels,
            self.feat_channels,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
            deform_groups=self.deform_groups,
            bias=False)

        self.relu = build_activation_layer(dict(type='ReLU'))

    def init_weights(self):
        """Initialize weights of the head."""
        nn.init.constant_(self.factor.weight, 0.5)
        normal_init(self.deform_conv, std=0.01)

    def forward_single(self, feats, anchor, score, stride, factor):
        """Forward function for single level."""
        num_imgs, _, H, W = feats[0].shape
        offset_list = [
            self.get_offset(anchor[i].reshape(-1, 5), (H, W), stride)
            for i in range(num_imgs)
        ]
        offset_tensor_3x3 = torch.stack(offset_list, dim=0)
        score = score.detach().max(dim=1, keepdim=True)[0]

        out_feats = tuple()
        for fn, x in enumerate(feats):
            _factor = factor[fn]

            score_1x1 = score.sigmoid() * _factor + (1 - _factor)
            score_3x3 = score_1x1.repeat(1, 9, 1, 1)

            # if self.deploy:
            out = self.deform_conv(x, offset_tensor_3x3, score_3x3)

            out = self.relu(out)
            out_feats = out_feats + (out, )
        return out_feats

    def forward(self, x: List[Tuple[Tensor]], anchors: List[List[Tensor]],
                scores: List[Tensor]) -> List[Tensor]:
        """Forward function."""
        level = len(x)
        mlvl_anchors = []
        for i in range(level):
            anchor = torch.stack([anchor[i] for anchor in anchors], dim=0)
            mlvl_anchors.append(anchor)

        factors = self.factor.weight.clone().clamp(min=0, max=1)
        # factors = self.factor.weight.clone().sigmoid()

        out = []
        for i, (feats, anchor, score, stride) in enumerate(
                zip(x, mlvl_anchors, scores, self.strides)):
            factor = factors[i]
            out.append(
                self.forward_single(feats, anchor, score, stride, factor))

        return out


@TASK_UTILS.register_module()
class PseudoAlignModule(BaseModule):
    """Pseudo Align Module."""

    def forward(self, x: List[Tensor],
                anchors: List[List[Tensor]]) -> List[Tensor]:
        """Forward function."""
        return x


@TASK_UTILS.register_module()
class DCNAlignModule(DeformConv2dPack):
    """DCN Align Module.

    All args are from DeformConv2dPack.
    TODO: maybe use build_conv_layer is more flexible.
    """

    def forward(self, x: List[Tensor],
                anchors: List[List[Tensor]]) -> List[Tensor]:
        """Forward function."""
        return [super(DCNAlignModule, self).forward(xi) for xi in x]


@TASK_UTILS.register_module()
class FRM(BaseModule):
    """Feature refine module for `R3Det`.

    Args:
        feat_channels (int): Number of input channels.
        strides (list[int]): The strides of featmap.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
    """

    def __init__(self, feat_channels: int, strides: List[int]) -> None:
        super().__init__()
        self.feat_channels = feat_channels
        self.strides = strides
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of feature refine module."""
        self.conv_5_1 = nn.Conv2d(
            in_channels=self.feat_channels,
            out_channels=self.feat_channels,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0))
        self.conv_1_5 = nn.Conv2d(
            in_channels=self.feat_channels,
            out_channels=self.feat_channels,
            kernel_size=(1, 5),
            stride=1,
            padding=(0, 2))
        self.conv_1_1 = nn.Conv2d(
            in_channels=self.feat_channels,
            out_channels=self.feat_channels,
            kernel_size=1)

    def init_weights(self) -> None:
        """Initialize weights of feature refine module."""
        normal_init(self.conv_5_1, std=0.01)
        normal_init(self.conv_1_5, std=0.01)
        normal_init(self.conv_1_1, std=0.01)

    def forward(self, x: List[Tensor],
                anchors: List[List[Tensor]]) -> List[Tensor]:
        """Forward function.

        Args:
            x (list[Tensor]): feature maps of multiple scales
            anchors (list[list[Tensor]]): anchors of multiple
                scales of multiple images

        Returns:
            list[Tensor]: refined feature maps of multiple scales.
        """
        mlvl_rbboxes = [torch.cat(best_rbbox) for best_rbbox in zip(*anchors)]
        out = []
        for x_scale, best_rbboxes_scale, fr_scale in zip(
                x, mlvl_rbboxes, self.strides):
            feat_scale_1 = self.conv_5_1(self.conv_1_5(x_scale))
            feat_scale_2 = self.conv_1_1(x_scale)
            feat_scale = feat_scale_1 + feat_scale_2
            feat_refined_scale = rotated_feature_align(feat_scale,
                                                       best_rbboxes_scale,
                                                       1 / fr_scale)
            out.append(x_scale + feat_refined_scale)
        return out
