# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch


class AngelCoder:
    """Angel Coder for Angel Coded Method such as CSL and DCL.

    This coder encodes angel into coded label and decodes
    coded label back to original angel.

    Args:
        type:
        angle_version:
        category_deg:
        window:
        radius:
    """

    def __init__(self,
                 type,
                 angle_version,
                 omega=1,
                 window='gaussian',
                 radius=6):
        self.type = type
        self.angle_version = angle_version
        assert angle_version in ['oc', 'le90', 'le135']
        self.angle_range = 90 if angle_version == 'oc' else 180
        self.angle_offset = 45 if angle_version == 'le135' else 90
        self.omega = omega
        self.window = window
        self.radius = radius
        if self.type == 'csl':
            self.coding_len = int(self.angle_range // omega)
        else:
            raise NotImplementedError

    # @property
    # def coding_length(self):

    def encode(self, angel_targets):
        if self.type == 'csl':
            return self.csl_encode(angel_targets)
        elif self.type == 'dcl':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def decode(self, angel_preds):
        if self.type == 'csl':
            return self.csl_decode(angel_preds)
        elif self.type == 'dcl':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def csl_encode(self, angle_targets):
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

    def csl_decode(self, angel_preds):
        angle_cls_inds = torch.argmax(angel_preds, dim=1)
        angle_pred = ((angle_cls_inds + 0.5) *
                      self.omega) % self.angle_range - self.angle_offset
        return angle_pred * (math.pi / 180)
