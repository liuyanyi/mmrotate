# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
from mmcv.cnn import (PLUGIN_LAYERS, ConvModule, build_activation_layer,
                      constant_init, normal_init)
from mmcv.runner import BaseModule
from mmdet.models.necks.dyhead import DyDCNv2
from torch import nn
from torch.nn import functional as F

from mmrotate.models.builder import ROTATED_NECKS


class ECA(nn.Module):

    def __init__(self, in_channels):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        kernel_size = np.ceil(0.5 * (np.log2(in_channels) + 1)).astype(int)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        mmcv.get_logger('ECA').info('kernel_size: {}'.format(kernel_size))
        mmcv.get_logger('ECA').info('in_channels: {}'.format(in_channels))
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding='same', bias=False)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1,
                                              -2)).transpose(-1,
                                                             -2).unsqueeze(-1)
        return x * self.sigmoid(y) * 2


@PLUGIN_LAYERS.register_module()
class CSA(nn.Module):

    def __init__(self, in_channels=None):
        super(CSA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Hardsigmoid()
        self.wq = nn.Parameter(torch.ones(1, 1, 1))
        self.wk = nn.Parameter(torch.ones(1, 1, 1))
        self.wv = nn.Parameter(torch.ones(1, 1, 1))

    def init_weights(self):
        normal_init(self.wq, std=0.01)
        normal_init(self.wk, std=0.01)
        normal_init(self.wv, std=0.01)

    def forward(self, x):
        # (b, c, h, w) -> (b, c, 1, 1)
        y = self.avg_pool(x)
        # (b, c, 1, 1) -> (b, 1, c)
        y = y.squeeze(-1).transpose(-1, -2)

        q, k, v = self.wq * y, self.wk * y, self.wv * y

        # (b, 1, c) -> (b, c, c)
        a = torch.bmm(k.transpose(1, 2), q)
        a = F.softmax(a, dim=-1)

        # (b, c, c) -> (b, 1, c)
        o = torch.bmm(v, a)

        # (b, 1, c) -> (b, c, 1, 1)
        o = o.transpose(1, 2).unsqueeze(-1)

        return x * self.sigmoid(o)


@PLUGIN_LAYERS.register_module()
class DCA(nn.Module):

    def __init__(self, in_channels=None):
        super(DCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Hardsigmoid()
        kernel_size = np.ceil(0.5 * (np.log2(in_channels) - 1)).astype(int)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        mmcv.get_logger('DCA').info('kernel_size: {}'.format(kernel_size))
        mmcv.get_logger('DCA').info('in_channels: {}'.format(in_channels))
        self.conv = nn.Conv1d(
            1, 1, kernel_size=int(kernel_size), padding='same', bias=False)
        self.dconv = nn.Conv1d(
            1,
            1,
            kernel_size=int(kernel_size),
            padding='same',
            dilation=2,
            bias=False)

    def forward(self, x):
        # (b, c, h, w) -> (b, c, 1, 1)
        y = self.avg_pool(x)
        # (b, c, 1, 1) -> (b, 1, c)
        y = y.squeeze(-1).transpose(-1, -2)

        y_l = self.conv(y).transpose(-1, -2)
        y_r = self.dconv(y).transpose(-1, -2)

        y = y_l + y_r
        y = y / 2
        y = y.unsqueeze(-1)

        return x * self.sigmoid(y) * 2


@PLUGIN_LAYERS.register_module()
class DSA(nn.Module):

    def __init__(self, in_channels, kernel_size=3):
        super(DSA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # kernel_size = np.ceil(0.5 * (np.log2(in_channels) - 1)).astype(int)
        # kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        mmcv.get_logger('DSA').info('kernel_size: {}'.format(kernel_size))
        mmcv.get_logger('DSA').info('in_channels: {}'.format(in_channels))
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding='same', bias=False)
        self.dconv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding='same',
            dilation=2,
            bias=False)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        y1 = x.permute(0, 2, 1, 3)
        y2 = x.permute(0, 3, 1, 2)

        y1 = self.avg_pool(y1).squeeze(-1)
        y2 = self.avg_pool(y2).squeeze(-1)

        y1_l = self.conv(y1.transpose(-1, -2)).transpose(-1, -2)
        y1_r = self.dconv(y1.transpose(-1, -2)).transpose(-1, -2)

        y1 = y1_l + y1_r
        y1 = y1 / 2

        y2_l = self.conv(y2.transpose(-1, -2)).transpose(-1, -2)
        y2_r = self.dconv(y2.transpose(-1, -2)).transpose(-1, -2)

        y2 = y2_l + y2_r
        y2 = y2 / 2
        y = torch.bmm(y1, y2.transpose(-1, -2)).unsqueeze(1)

        return x * self.sigmoid(y) * 2


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 16*1*32*32
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 16*1*32*32
        x1 = torch.cat([avg_out, max_out], dim=1)  # 16*2*32*32
        x2 = self.conv1(x1)  # 16*1*32*32
        x3 = self.sigmoid(x2)
        out_s_a = x * x3
        return out_s_a


class ScaleBlockBase(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super().__init__()
        norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
        self.high_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg)
        self.mid_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg)
        self.low_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, 1, 1),
            nn.ReLU(inplace=True), build_activation_layer(act_cfg))
        self.channel_attn_module = DCA(in_channels)
        self.spatial_attn_module = SpatialAttention()

    def forward_scale(self, x):
        outs = []
        for level in range(len(x)):
            mid_feat = self.mid_conv(x[level])
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = self.low_conv(x[level - 1])
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    self.high_conv(x[level + 1]),
                    size=x[level].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            sum_feat = sum_feat / summed_levels
            outs.append(sum_feat)
        return outs

    def forward_channel(self, x):
        return [self.channel_attn_module(level) for level in x]

    def forward_spatial(self, x):
        return [self.spatial_attn_module(level) for level in x]

    def forward(self, x):
        """Forward function."""
        outs = []
        for level in range(len(x)):

            mid_feat = x[level]
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = F.interpolate(
                    x[level - 1],
                    size=x[level].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    x[level + 1],
                    size=x[level].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            sum_feat = sum_feat / summed_levels
            sum_feat = self.spatial_attn_module(sum_feat)
            sum_feat = self.channel_attn_module(sum_feat)
            outs.append(sum_feat)

        return outs


class TA(ScaleBlockBase):

    def forward(self, x):
        x = self.forward_scale(x)
        x = self.forward_channel(x)
        x = self.forward_spatial(x)
        return x


class TB(ScaleBlockBase):

    def forward(self, x):
        x = self.forward_scale(x)
        x = self.forward_spatial(x)
        x = self.forward_channel(x)
        return x


class TC(ScaleBlockBase):

    def forward(self, x):
        x = self.forward_channel(x)
        x = self.forward_scale(x)
        x = self.forward_spatial(x)
        return x


class TD(ScaleBlockBase):

    def forward(self, x):
        x = self.forward_channel(x)
        x = self.forward_spatial(x)
        x = self.forward_scale(x)
        return x


class TE(ScaleBlockBase):

    def forward(self, x):
        x = self.forward_spatial(x)
        x = self.forward_channel(x)
        x = self.forward_scale(x)
        return x


class TF(ScaleBlockBase):

    def forward(self, x):
        x = self.forward_spatial(x)
        x = self.forward_scale(x)
        x = self.forward_channel(x)
        return x


class DyHeadBlock(nn.Module):
    """DyHead Block with three types of attention.

    HSigmoid arguments in default act_cfg follow official code, not paper.
    https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        act_cfg (dict, optional): Config dict for the last activation layer of
            scale-aware attention. Default: dict(type='HSigmoid', bias=3.0,
            divisor=6.0).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super().__init__()
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.offset_and_mask_dim = 3 * 3 * 3
        self.offset_dim = 2 * 3 * 3

        self.spatial_conv_high = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_mid = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_low = DyDCNv2(in_channels, out_channels, stride=2)
        self.spatial_conv_offset = nn.Conv2d(
            in_channels, self.offset_and_mask_dim, 3, padding=1)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels, 1, 1),
            nn.ReLU(inplace=True), build_activation_layer(act_cfg))
        self.task_attn_module = ECA(out_channels)

        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        constant_init(self.spatial_conv_offset, 0)

    def forward_scale(self, x):
        """Forward function."""
        outs = []
        for level in range(len(x)):
            # calculate offset and mask of DCNv2 from middle-level feature
            offset_and_mask = self.spatial_conv_offset(x[level])
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()

            mid_feat = self.spatial_conv_mid(x[level], offset, mask)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = self.spatial_conv_low(x[level - 1], offset, mask)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    self.spatial_conv_high(x[level + 1], offset, mask),
                    size=x[level].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(sum_feat / summed_levels)

        return outs

    def forward_task(self, x):
        return [self.task_attn_module(level) for level in x]

    def forward(self, x):
        x = self.forward_scale(x)
        x = self.forward_task(x)
        return x


class DyHeadBlockB(DyHeadBlock):

    def forward(self, x):
        x = self.forward_task(x)
        x = self.forward_scale(x)
        return x


class DyHeadBlockC(nn.Module):
    """DyHead Block with three types of attention.

    HSigmoid arguments in default act_cfg follow official code, not paper.
    https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        act_cfg (dict, optional): Config dict for the last activation layer of
            scale-aware attention. Default: dict(type='HSigmoid', bias=3.0,
            divisor=6.0).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super().__init__()
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.offset_and_mask_dim = 3 * 3 * 3
        self.offset_dim = 2 * 3 * 3

        self.spatial_conv_high = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_mid = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_low = DyDCNv2(in_channels, out_channels, stride=2)
        self.spatial_conv_offset = nn.Conv2d(
            in_channels, self.offset_and_mask_dim, 3, padding=1)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels, 1, 1),
            nn.ReLU(inplace=True), build_activation_layer(act_cfg))
        self.front_task_attn_module = ECA(out_channels)
        self.last_task_attn_module = ECA(out_channels)

        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        constant_init(self.spatial_conv_offset, 0)

    def forward_scale(self, x):
        """Forward function."""
        outs = []
        for level in range(len(x)):
            # calculate offset and mask of DCNv2 from middle-level feature
            offset_and_mask = self.spatial_conv_offset(x[level])
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()

            mid_feat = self.spatial_conv_mid(x[level], offset, mask)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = self.spatial_conv_low(x[level - 1], offset, mask)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    self.spatial_conv_high(x[level + 1], offset, mask),
                    size=x[level].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(sum_feat / summed_levels)

        return outs

    def forward_task(self, x, module):
        return [module(level) for level in x]

    def forward(self, x):
        x = self.forward_task(x, self.front_task_attn_module)
        x = self.forward_scale(x)
        x = self.forward_task(x, self.last_task_attn_module)
        return x


class HFA(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super().__init__()
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.offset_and_mask_dim = 3 * 3 * 3
        self.offset_dim = 2 * 3 * 3

        self.spatial_conv_high = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_mid = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_low = DyDCNv2(in_channels, out_channels, stride=2)
        self.spatial_conv_offset = nn.Conv2d(
            in_channels, self.offset_and_mask_dim, 3, padding=1)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels, 1, 1),
            nn.ReLU(inplace=True), build_activation_layer(act_cfg))
        self.task_attn_module = ECA(out_channels)

        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        constant_init(self.spatial_conv_offset, 0)

    def forward_scale(self, x):
        """Forward function."""
        outs = []
        for level in range(len(x)):
            # calculate offset and mask of DCNv2 from middle-level feature
            offset_and_mask = self.spatial_conv_offset(x[level])
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()

            mid_feat = self.spatial_conv_mid(x[level], offset, mask)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = self.spatial_conv_low(x[level - 1], offset, mask)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    self.spatial_conv_high(x[level + 1], offset, mask),
                    size=x[level].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(sum_feat / summed_levels)

        return outs

    def forward_task(self, x):
        return [self.task_attn_module(level) for level in x]

    def forward(self, x):
        x = self.forward_task(x)
        x = self.forward_scale(x)
        return x


class HFAB(HFA):

    def forward(self, x):
        x = self.forward_scale(x)
        x = self.forward_task(x)
        return x


class HFA_E(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super().__init__()
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        # self.offset_and_mask_dim = 3 * 3 * 3
        # self.offset_dim = 2 * 3 * 3
        #
        # self.spatial_conv_high = DyDCNv2(in_channels, out_channels)
        # self.spatial_conv_mid = DyDCNv2(in_channels, out_channels)
        # self.spatial_conv_low = DyDCNv2(in_channels, out_channels, stride=2)
        # self.spatial_conv_offset = nn.Conv2d(
        #     in_channels, self.offset_and_mask_dim, 3, padding=1)
        # self.scale_attn_module = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels, 1, 1),
        #     nn.ReLU(inplace=True), build_activation_layer(act_cfg))
        self.task_attn_module = ECA(out_channels)

        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        # constant_init(self.spatial_conv_offset, 0)

    def forward_scale(self, x):
        """Forward function."""
        outs = []
        for level in range(len(x)):
            # calculate offset and mask of DCNv2 from middle-level feature
            offset_and_mask = self.spatial_conv_offset(x[level])
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()

            mid_feat = self.spatial_conv_mid(x[level], offset, mask)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = self.spatial_conv_low(x[level - 1], offset, mask)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    self.spatial_conv_high(x[level + 1], offset, mask),
                    size=x[level].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(sum_feat / summed_levels)

        return outs

    def forward_task(self, x):
        return [self.task_attn_module(level) for level in x]

    def forward(self, x):
        x = self.forward_task(x)
        # x = self.forward_scale(x)
        return x


class HFA_SS(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super().__init__()
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.offset_and_mask_dim = 3 * 3 * 3
        self.offset_dim = 2 * 3 * 3

        self.spatial_conv_high = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_mid = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_low = DyDCNv2(in_channels, out_channels, stride=2)
        self.spatial_conv_offset = nn.Conv2d(
            in_channels, self.offset_and_mask_dim, 3, padding=1)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels, 1, 1),
            nn.ReLU(inplace=True), build_activation_layer(act_cfg))
        # self.task_attn_module = ECA(out_channels)

        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        constant_init(self.spatial_conv_offset, 0)

    def forward_scale(self, x):
        """Forward function."""
        outs = []
        for level in range(len(x)):
            # calculate offset and mask of DCNv2 from middle-level feature
            offset_and_mask = self.spatial_conv_offset(x[level])
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()

            mid_feat = self.spatial_conv_mid(x[level], offset, mask)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = self.spatial_conv_low(x[level - 1], offset, mask)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    self.spatial_conv_high(x[level + 1], offset, mask),
                    size=x[level].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(sum_feat / summed_levels)

        return outs

    def forward_task(self, x):
        return [self.task_attn_module(level) for level in x]

    def forward(self, x):
        # x = self.forward_task(x)
        x = self.forward_scale(x)
        return x


@ROTATED_NECKS.register_module()
class ScaleFusion(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 name,
                 num_blocks=1,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        blocks = []
        for i in range(num_blocks):
            in_channels = self.in_channels if i == 0 else self.out_channels
            blocks.append(eval(name)(in_channels, out_channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs):
        """Forward function."""
        assert isinstance(inputs, (tuple, list))
        outs = self.blocks(inputs)
        return tuple(outs)


if __name__ == '__main__':
    a = torch.rand(1, 160, 256, 256)
    deca = ScaleFusion(160, 'TA')
    print(deca(a).shape)
