# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder

from ..builder import ROTATED_BBOX_CODERS


@ROTATED_BBOX_CODERS.register_module()
class ACLCoder(BaseBBoxCoder):

    def __init__(self, angle_version):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['le90']
        self.angle_range = 90
        self.coding_len = 8
        self.coding_class_num = np.power(2, self.coding_len - 1)
        self.omega = self.angle_range / (self.coding_class_num - 1)

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
        sign = angle_targets_deg > 0
        angle_targets_deg = angle_targets_deg.abs()
        # empty label
        angle_targets_deg = angle_targets_deg / self.omega
        # Float to Int
        angle_targets_long = angle_targets_deg.long()
        assert angle_targets_long.min() >= 0
        assert angle_targets_long.max() < self.coding_class_num
        labels = []
        for i in range(self.coding_len - 1):
            labels.append(angle_targets_long % 2)
            angle_targets_long = angle_targets_long >> 1
        labels.append(sign.long())
        label = torch.cat(labels, dim=1)

        return label

    def decode(self, angle_preds):
        """Circular Smooth Label Decoder.

        Args:
            angle_preds (Tensor): The csl encoding of angle offset
                for each scale level.
                Has shape (num_anchors * H * W, coding_len)

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1)
        """
        sign = angle_preds[:, 0].sigmoid() > 0.5
        sign = sign.long() * 2 - 1
        angle_preds = angle_preds[:, 1:].sigmoid() >= 0.5
        angle_cls_inds = angle_preds.new_zeros(angle_preds.shape[0])
        for i in range(self.coding_len - 1):
            angle_cls_inds = angle_cls_inds + angle_preds[:, i].long(
            ) * np.power(2, i)

        angle_pred = ((angle_cls_inds + 0.5) * self.omega) % self.angle_range

        return angle_pred * (math.pi / 180) * sign
