# Copyright (c) OpenMMLab. All rights reserved.
import math
from abc import ABCMeta, abstractmethod

import torch
from mmcv import Registry, build_from_cfg
from torch import nn

ANGLE_CODER = Registry('angle_coder')


def build_angle_coder(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, ANGLE_CODER, default_args)


class BaseAngleCoder(metaclass=ABCMeta):
    """Base angle coder."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def encode(self, angle_targets):
        """Encode deltas between bboxes and ground truth boxes."""

    @abstractmethod
    def decode(self, angle_preds):
        """Decode the predicted bboxes according to prediction and base
        boxes."""


@ANGLE_CODER.register_module()
class CSLCoder(BaseAngleCoder):

    def __init__(self, angle_version, omega=1, window='gaussian', radius=6):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['oc', 'le90', 'le135']
        self.angle_range = 90 if angle_version == 'oc' else 180
        self.angle_offset = 45 if angle_version == 'le135' else 90
        self.omega = omega
        self.window = window
        self.radius = radius
        self.coding_len = int(self.angle_range // omega)

    def encode(self, angle_targets):
        # Radius To Degree
        angle_targets_deg = angle_targets * (180 / math.pi)
        # Empty Label
        smooth_label = torch.zeros_like(angle_targets).repeat(
            1, self.coding_len)
        angle_targets_deg = (angle_targets_deg +
                             self.angle_offset) / self.omega
        # TODO 要不要四舍五入，这里是直接舍掉，解码再+0.5
        angle_targets_long = angle_targets_deg.long()

        if self.window == 'pulse':
            radius_range = angle_targets_long
            smooth_value = 1.0
        elif self.window == 'rect':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.coding_len
            smooth_value = 1.0
        elif self.window == 'triangle':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.coding_len
            smooth_value = 1.0 - torch.abs(
                (1 / self.radius) * base_radius_range)

        elif self.window == 'gaussian':
            base_radius_range = torch.arange(
                -self.angle_range // 2,
                self.angle_range // 2,
                device=angle_targets_long.device)

            radius_range = (base_radius_range +
                            angle_targets_long) % self.coding_len
            smooth_value = torch.exp(-torch.pow(base_radius_range, 2) /
                                     (2 * self.radius**2))

        else:
            raise NotImplementedError

        # 调整维度
        if isinstance(smooth_value, torch.Tensor):
            smooth_value = smooth_value.unsqueeze(0).repeat(
                smooth_label.size(0), 1)

        return smooth_label.scatter(1, radius_range, smooth_value)

    def decode(self, angle_preds):
        angle_cls_inds = torch.argmax(angle_preds, dim=1)
        angle_pred = ((angle_cls_inds + 0.5) *
                      self.omega) % self.angle_range - self.angle_offset
        return angle_pred * (math.pi / 180)

    def soft_decode(self, angle_preds):
        angle_cls_inds = self.softargmax1d(angle_preds)
        angle_pred = ((angle_cls_inds + 0.5) *
                      self.omega) % self.angle_range - self.angle_offset
        return angle_pred * (math.pi / 180)

    def softargmax1d(self, input, beta=5000):
        device = input.device
        *_, n = input.shape
        input = nn.functional.softmax(beta * input, dim=-1)
        indices = torch.linspace(0, 1, n, device=device)
        result = torch.sum((n - 1) * input * indices, dim=-1)
        return result


@ANGLE_CODER.register_module()
class BCLCoder(BaseAngleCoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode(self, angle_targets):
        pass

    def decode(self, angle_preds):
        pass


@ANGLE_CODER.register_module()
class GCLCoder(BaseAngleCoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode(self, angle_targets):
        pass

    def decode(self, angle_preds):
        pass
