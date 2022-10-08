# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder

from mmrotate.registry import TASK_UTILS


@TASK_UTILS.register_module()
class CSLCoder(BaseBBoxCoder):
    """Circular Smooth Label Coder.

    `Circular Smooth Label (CSL)
    <https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40>`_ .

    Args:
        angle_version (str): Angle definition.
        omega (float, optional): Angle discretization granularity.
            Default: 1.
        window (str, optional): Window function. Default: gaussian.
        radius (int/float): window radius, int type for
            ['triangle', 'rect', 'pulse'], float type for
            ['gaussian']. Default: 6.
    """

    def __init__(self, angle_version, omega=1, window='gaussian', radius=6):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['oc', 'le90', 'le135']
        assert window in ['gaussian', 'triangle', 'rect', 'pulse']
        self.angle_range = 90 if angle_version == 'oc' else 180
        self.angle_offset_dict = {'oc': 0, 'le90': 90, 'le135': 45}
        self.angle_offset = self.angle_offset_dict[angle_version]
        self.omega = omega
        self.window = window
        self.radius = radius
        self.encode_size = int(self.angle_range // omega)

    def encode(self, angle_targets):
        """Circular Smooth Label Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level
                Has shape (num_anchors * H * W, 1)

        Returns:
            list[Tensor]: The csl encoding of angle offset for each
                scale level. Has shape (num_anchors * H * W, coding_len)
        """

        # radius to degree
        angle_targets_deg = angle_targets * (180 / math.pi)
        # empty label
        smooth_label = torch.zeros_like(angle_targets).repeat(
            1, self.encode_size)
        angle_targets_deg = (angle_targets_deg +
                             self.angle_offset) / self.omega
        # Float to Int
        angle_targets_long = angle_targets_deg.long()

        if self.window == 'pulse':
            radius_range = angle_targets_long % self.encode_size
            smooth_value = 1.0
        elif self.window == 'rect':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.encode_size
            smooth_value = 1.0
        elif self.window == 'triangle':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.encode_size
            smooth_value = 1.0 - torch.abs(
                (1 / self.radius) * base_radius_range)

        elif self.window == 'gaussian':
            base_radius_range = torch.arange(
                -self.angle_range // 2,
                self.angle_range // 2,
                device=angle_targets_long.device)

            radius_range = (base_radius_range +
                            angle_targets_long) % self.encode_size
            smooth_value = torch.exp(-torch.pow(base_radius_range, 2) /
                                     (2 * self.radius**2))

        else:
            raise NotImplementedError

        if isinstance(smooth_value, torch.Tensor):
            smooth_value = smooth_value.unsqueeze(0).repeat(
                smooth_label.size(0), 1)

        return smooth_label.scatter(1, radius_range, smooth_value)

    def decode(self, angle_preds):
        """Circular Smooth Label Decoder.

        Args:
            angle_preds (Tensor): The csl encoding of angle offset
                for each scale level.
                Has shape (N, coding_len) or (B, N, coding_len)

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (N, 1) or (B, N, coding_len)
        """
        angle_cls_inds = torch.argmax(angle_preds, dim=-1, keepdim=True)
        angle_pred = ((angle_cls_inds + 0.5) *
                      self.omega) % self.angle_range - self.angle_offset
        return angle_pred * (math.pi / 180)


@TASK_UTILS.register_module()
class PseudoAngleCoder(BaseBBoxCoder):
    """"""

    encode_size = 1

    def __init__(self):
        super().__init__()
        self.coding_len = 1

    def encode(self, angle_targets):
        return angle_targets

    def decode(self, angle_preds):
        return angle_preds


@TASK_UTILS.register_module()
class DCLCoder(BaseBBoxCoder):

    def __init__(self, angle_version, encode_size=7, pos_thr=0.5):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['oc', 'le90', 'le135']
        self.angle_range = 90 if angle_version == 'oc' else 180
        self.angle_offset_dict = {'oc': 0, 'le90': 90, 'le135': 45}
        self.angle_offset = self.angle_offset_dict[angle_version]
        assert isinstance(encode_size, int)
        self.encode_size = encode_size
        self.omega = 180 / np.power(2, encode_size)
        self.pos_thr = pos_thr

    def encode(self, angle_targets):
        # radius to degree
        angle_targets_deg = angle_targets * (180 / math.pi)
        # empty label
        smooth_label = torch.zeros_like(angle_targets).repeat(
            1, self.encode_size)
        angle_targets_deg = (angle_targets_deg +
                             self.angle_offset) / self.omega
        # Float to Int
        angle_targets_long = angle_targets_deg.long()
        # print(angle_targets_long[0])

        # len~1
        for i in range(self.encode_size, 0, -1):
            smooth_label[:, i - 1:i] = angle_targets_long % 2
            angle_targets_long = angle_targets_long >> 1

        # print(smooth_label[0])
        return smooth_label

    def decode(self, angle_preds):
        power_value = np.power(2, range(self.encode_size - 1, -1, -1))
        power_value = torch.tensor(
            power_value, device=angle_preds.device, dtype=torch.long)

        postive_ind = angle_preds.sigmoid() > self.pos_thr

        angle_preds_bin = torch.zeros_like(angle_preds)
        angle_preds_bin[postive_ind] = 1

        angle_cls_inds = (angle_preds_bin * power_value).sum(
            keepdim=True, dim=-1)
        angle_pred = ((angle_cls_inds + 0.5) *
                      self.omega) % self.angle_range - self.angle_offset
        return angle_pred * (math.pi / 180)


@TASK_UTILS.register_module()
class DistributionAngleCoder(BaseBBoxCoder):

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def encode(self, angle):
        return self.reg_max * (0.5 * np.pi + angle) / np.pi

    def decode(self, angle):
        return np.pi * angle / self.reg_max - 0.5 * np.pi


@TASK_UTILS.register_module()
class DistributionScaleAngleCoder(BaseBBoxCoder):

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.start = np.array([0, 0.25, 0.5, 1])
        self.factor = np.array([2, 1, 0.5])
        self.target_start = np.array([0, 0.5, 0.75, 1])

    def encode(self, angle):
        if angle.dim() == 2:
            flatten = True
            angle = angle.flatten()
        else:
            flatten = False
        if isinstance(self.start, np.ndarray):
            self.start = angle.new_tensor(self.start)
            self.factor = angle.new_tensor(self.factor)
            self.target_start = angle.new_tensor(self.target_start)
        # To -1~1
        sig_code = angle.sign()
        norm_a = (2 * angle / np.pi).abs()

        seg_code = torch.div(
            norm_a[:, None], self.start[None, 1:],
            rounding_mode='trunc').clamp(
                min=-1, max=1).sum(dim=-1)
        seg_code = seg_code.long()
        factors = self.factor[seg_code]
        start = self.start[seg_code]
        target_start = self.target_start[seg_code]

        encoded_angle = sig_code * ((norm_a - start) * factors + target_start)
        # 范围调整(0~reg_max)
        encoded_angle = 0.5 * self.reg_max * (encoded_angle + 1)

        if flatten:
            encoded_angle = encoded_angle[:, None]
        return encoded_angle

    def decode(self, angle):
        if angle.dim() == 2:
            flatten = True
            angle = angle.flatten()
        else:
            flatten = False
        if isinstance(self.start, np.ndarray):
            self.start = angle.new_tensor(self.start)
            self.factor = angle.new_tensor(self.factor)
            self.target_start = angle.new_tensor(self.target_start)

        # To -1~1
        norm_a = (2 * angle / self.reg_max - 1)
        sig_code = norm_a.sign()
        norm_a = norm_a.abs()

        seg_code = torch.div(
            norm_a[:, None],
            self.target_start[None, 1:],
            rounding_mode='trunc').clamp(max=1).sum(dim=-1)
        seg_code = seg_code.long()
        factors = 1 / self.factor[seg_code]
        start = self.target_start[seg_code]
        target_start = self.start[seg_code]

        encoded_angle = sig_code * ((norm_a - start) * factors + target_start)
        encoded_angle = encoded_angle * 0.5 * np.pi

        if flatten:
            encoded_angle = encoded_angle[:, None]
        return encoded_angle


if __name__ == '__main__':
    coder = DistributionScaleAngleCoder(reg_max=16)
    ang = torch.Tensor([-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9
                        ]) * 0.5 * np.pi

    encoded = coder.encode(ang)
    decoded = coder.decode(encoded)

    print(ang)
    print(encoded)
    print(decoded)
