# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmdet.models import SELayer
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
from mmengine.model import BaseModule, Sequential
from torch import Tensor

from mmrotate.registry import MODELS


class RepConvModule(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 se_cfg=None,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 deploy=False,
                 init_cfg=None):
        super(RepConvModule, self).__init__(init_cfg)

        assert se_cfg is None or isinstance(se_cfg, dict)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.se_cfg = se_cfg
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.deploy = deploy

        if deploy:
            self.branch_reparam = build_conv_layer(
                conv_cfg,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode)
        else:
            # judge if input shape and output shape are the same.
            # If true, add a normalized identity shortcut.
            if out_channels == in_channels and stride == 1 and \
                    padding == dilation:
                self.branch_norm = build_norm_layer(norm_cfg, in_channels)[1]
            else:
                self.branch_norm = None

            self.branch_3x3 = self.create_conv_bn(
                kernel_size=3,
                dilation=dilation,
                padding=padding,
            )
            self.branch_1x1 = self.create_conv_bn(kernel_size=1)

        if se_cfg is not None:
            self.se_layer = SELayer(channels=out_channels, **se_cfg)
        else:
            self.se_layer = None

        self.act = build_activation_layer(act_cfg)

    def create_conv_bn(self, kernel_size, dilation=1, padding=0):
        conv_bn = Sequential()
        conv_bn.add_module(
            'conv',
            build_conv_layer(
                self.conv_cfg,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                dilation=dilation,
                padding=padding,
                groups=self.groups,
                bias=False))
        conv_bn.add_module(
            'norm',
            build_norm_layer(self.norm_cfg, num_features=self.out_channels)[1])

        return conv_bn

    def forward(self, x):

        def _inner_forward(inputs):
            if self.deploy:
                return self.branch_reparam(inputs)

            if self.branch_norm is None:
                branch_norm_out = 0
            else:
                branch_norm_out = self.branch_norm(inputs)

            inner_out = self.branch_3x3(inputs) + self.branch_1x1(
                inputs) + branch_norm_out

            if self.se_cfg is not None:
                inner_out = self.se_layer(inner_out)

            return inner_out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.act(out)

        return out

    def switch_to_deploy(self):
        """Switch the model structure from training mode to deployment mode."""
        if self.deploy:
            return
        assert self.norm_cfg['type'] == 'BN', \
            "Switch is not allowed when norm_cfg['type'] != 'BN'."

        reparam_weight, reparam_bias = self.reparameterize()
        self.branch_reparam = build_conv_layer(
            self.conv_cfg,
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True)
        self.branch_reparam.weight.data = reparam_weight
        self.branch_reparam.bias.data = reparam_bias

        for param in self.parameters():
            param.detach_()
        delattr(self, 'branch_3x3')
        delattr(self, 'branch_1x1')
        delattr(self, 'branch_norm')

        self.deploy = True

    def reparameterize(self):
        """Fuse all the parameters of all branches.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Parameters after fusion of all
                branches. the first element is the weights and the second is
                the bias.
        """
        weight_3x3, bias_3x3 = self._fuse_conv_bn(self.branch_3x3)
        weight_1x1, bias_1x1 = self._fuse_conv_bn(self.branch_1x1)
        # pad a conv1x1 weight to a conv3x3 weight
        weight_1x1 = F.pad(weight_1x1, [1, 1, 1, 1], value=0)

        weight_norm, bias_norm = 0, 0
        if self.branch_norm:
            tmp_conv_bn = self._norm_to_conv3x3(self.branch_norm)
            weight_norm, bias_norm = self._fuse_conv_bn(tmp_conv_bn)

        return (weight_3x3 + weight_1x1 + weight_norm,
                bias_3x3 + bias_1x1 + bias_norm)

    def _fuse_conv_bn(self, branch):
        """Fuse the parameters in a branch with a conv and bn.

        Args:
            branch (mmcv.runner.Sequential): A branch with conv and bn.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The parameters obtained after
                fusing the parameters of conv and bn in one branch.
                The first element is the weight and the second is the bias.
        """
        if branch is None:
            return 0, 0
        conv_weight = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps

        std = (running_var + eps).sqrt()
        fused_weight = (gamma / std).reshape(-1, 1, 1, 1) * conv_weight
        fused_bias = -running_mean * gamma / std + beta

        return fused_weight, fused_bias

    def _norm_to_conv3x3(self, branch_nrom):
        """Convert a norm layer to a conv3x3-bn sequence.

        Args:
            branch (nn.BatchNorm2d): A branch only with bn in the block.
        Returns:
            tmp_conv3x3 (mmcv.runner.Sequential): a sequential with conv3x3 and
                bn.
        """
        input_dim = self.in_channels // self.groups
        conv_weight = torch.zeros((self.in_channels, input_dim, 3, 3),
                                  dtype=branch_nrom.weight.dtype)

        for i in range(self.in_channels):
            conv_weight[i, i % input_dim, 1, 1] = 1
        conv_weight = conv_weight.to(branch_nrom.weight.device)

        tmp_conv3x3 = self.create_conv_bn(kernel_size=3)
        tmp_conv3x3.conv.weight.data = conv_weight
        tmp_conv3x3.norm = branch_nrom
        return tmp_conv3x3


@MODELS.register_module()
class RepFPN(BaseModule):

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        deploy: bool = False,
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.deploy = deploy

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = RepConvModule(
                out_channels,
                out_channels,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                deploy=deploy)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = RepConvModule(
                    in_channels,
                    out_channels,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    deploy=deploy)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
