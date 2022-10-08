# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmdet.models import RetinaHead
from mmdet.models.dense_heads import FCOSHead
from mmdet.models.task_modules import (PseudoSampler, SimOTAAssigner,
                                       anchor_inside_flags)
from mmdet.models.utils import (filter_scores_and_topk, images_to_levels,
                                multi_apply, select_single_mlvl, unmap)
from mmdet.structures.bbox import (HorizontalBoxes, bbox_overlaps, cat_boxes,
                                   get_box_tensor)
from mmdet.utils import (ConfigType, InstanceList, MultiConfig,
                         OptInstanceList, RangeType, reduce_mean)
from mmengine import ConfigDict
from mmengine.hooks import Hook
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.core import (PseudoRotatedAnchorGenerator, RotatedBoxes,
                           norm_angle, rbbox_overlaps)
from mmrotate.core.bbox.assigners import RSimOTAAssigner
from mmrotate.models.utils import ORConv2d, RotationInvariantPooling
from mmrotate.registry import HOOKS, MODELS, TASK_UTILS
from .rotated_fcos_head import RotatedFCOSHead
from .s2a_head import WS2ARefineHead

INF = 1e8


@HOOKS.register_module()
class QQQHook(Hook):

    def __init__(self, start_epoch=6):
        self.start_epoch = start_epoch

    def before_train(self, runner) -> None:
        self.sv = runner.model.train_cfg.stage_loss_weights
        runner.model.train_cfg.stage_loss_weights = [0 for _ in self.sv]

    def before_train_epoch(self, runner) -> None:
        super().before_train_epoch(runner)
        if runner.epoch < self.start_epoch:
            runner.model.train_cfg.stage_loss_weights = [0 for _ in self.sv]
        else:
            runner.model.train_cfg.stage_loss_weights = self.sv


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: ``sum{P(y_i) * y_i}``,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Defaults to 16.
            You may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max: int = 10) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 1)
        return x


class IIII(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: ``sum{P(y_i) * y_i}``,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Defaults to 16.
            You may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max: int = 10) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.l_mov_size = 5
        self.r_mov_size = reg_max - 5 + 1
        base_proj = torch.linspace(0, self.reg_max, self.reg_max + 1)
        left_proj = torch.linspace(self.l_mov_size,
                                   self.reg_max + self.l_mov_size,
                                   self.reg_max + 1)
        right_proj = torch.linspace(self.r_mov_size,
                                    self.reg_max + self.r_mov_size,
                                    self.reg_max + 1)
        self.register_buffer('b_proj', base_proj)
        self.register_buffer('l_proj', left_proj)
        self.register_buffer('r_proj', right_proj)

    def forward(self, x: Tensor, s_out=False) -> Tensor:
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        b_x = F.linear(x, self.b_proj.type_as(x)).reshape(-1, 1)
        l_x = F.linear(x, self.l_proj.type_as(x)).reshape(-1, 1)
        r_x = F.linear(x, self.r_proj.type_as(x)).reshape(-1, 1)
        l_x = (l_x - self.l_mov_size) % self.reg_max
        r_x = (r_x - self.r_mov_size) % self.reg_max
        if self.training and not s_out:
            return [b_x, l_x, r_x]
        else:
            diff_1 = torch.abs(b_x - l_x)
            diff_2 = torch.abs(b_x - r_x)
            diff_3 = torch.abs(l_x - r_x)

            x_s = torch.cat([b_x + l_x, b_x + r_x, l_x + r_x], dim=-1) / 2

            diff = torch.stack([diff_1, diff_2, diff_3], dim=-1)
            min_d = torch.argmin(diff, dim=-1)

            return x_s[:, min_d]


@MODELS.register_module()
class QQQHead(RotatedFCOSHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 regress_ranges: RangeType = ((-1, 64), (64, 128), (128, 256),
                                              (256, 512), (512, INF)),
                 center_sampling: bool = False,
                 center_sample_radius: float = 1.5,
                 norm_on_bbox: bool = False,
                 loss_cls: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(type='IoULoss', loss_weight=1.0),
                 angle_version: str = 'le90',
                 separate_angle: bool = False,
                 scale_angle: bool = True,
                 angle_coder: ConfigType = dict(type='PseudoAngleCoder'),
                 h_bbox_coder: ConfigType = dict(
                     type='mmdet.DistancePointBBoxCoder'),
                 reg_max: int = 16,
                 loss_angle: ConfigType = dict(
                     type='mmdet.L1Loss', loss_weight=1.0),
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.angle_version = angle_version
        self.separate_angle = separate_angle
        self.is_scale_angle = scale_angle
        self.angle_coder = TASK_UTILS.build(angle_coder)
        self.reg_max = reg_max
        super(FCOSHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_angle = MODELS.build(loss_angle)
        if self.separate_angle:
            # self.loss_angle = MODELS.build(loss_angle)
            self.h_bbox_coder = TASK_UTILS.build(h_bbox_coder)
        self.integral = Integral(self.reg_max)

    def _init_layers(self):
        """Initialize layers of the head."""
        super(FCOSHead, self)._init_layers()
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.conv_angle = nn.Conv2d(
            # self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
            self.feat_channels,
            self.reg_max + 1,
            3,
            padding=1)
        if self.is_scale_angle:
            self.scale_angle = Scale(1.0)

    def forward(
            self,
            x: Tuple[Tensor],
            with_feats=False
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        return multi_apply(
            self.forward_single,
            x,
            self.scales,
            self.strides,
            with_feats=with_feats)

    def forward_single(self,
                       x: Tensor,
                       scale: Scale,
                       stride: int,
                       with_feats: bool = False
                       ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.
        Returns:
            tuple: scores for each class, bbox predictions, angle predictions \
                and centerness predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super(
            FCOSHead, self).forward_single(x)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        angle_pred = self.conv_angle(reg_feat)
        if self.is_scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()
        if with_feats:
            return cls_score, bbox_pred, angle_pred, cls_feat, reg_feat
        else:
            return cls_score, bbox_pred, angle_pred

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        angle_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            angle_preds (list[Tensor]): Box angle for each scale level, each \
                is a 4D-tensor, the channel number is num_points * encode_size.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets, angle_targets = self.get_targets(
            all_level_points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds, angle_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        angle_dim = self.angle_coder.encode_size
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, self.reg_max + 1)
            # angle_pred.permute(0, 2, 3, 1).reshape(-1, angle_dim)
            for angle_pred in angle_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        # pos_angle_targets = self.angle_coder.encode(pos_angle_targets)
        score = pos_bbox_targets.new_zeros(flatten_labels.shape)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            if self.separate_angle:
                bbox_coder = self.h_bbox_coder
                overlap_func = bbox_overlaps
            else:
                bbox_coder = self.bbox_coder
                # pos_angle_preds = self.angle_coder.decode(pos_angle_preds)
                poe = self.integral(pos_angle_preds)
                # pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
                # poe = np.pi * poe / 16 - 0.5 * np.pi
                poe = self.angle_coder.decode(poe)

                pos_bbox_preds = torch.cat([pos_bbox_preds, poe], dim=-1)
                pos_bbox_targets = torch.cat(
                    [pos_bbox_targets, pos_angle_targets], dim=-1)
                overlap_func = rbbox_overlaps
            pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
                                                       pos_bbox_preds)
            pos_decoded_bbox_targets = bbox_coder.decode(
                pos_points, pos_bbox_targets)

            weight_targets = flatten_cls_scores.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]

            score[pos_inds] = overlap_func(
                pos_decoded_bbox_preds.detach(),
                pos_decoded_bbox_targets,
                is_aligned=True)
            weight_denorm = max(
                reduce_mean(weight_targets.sum().detach()), 1e-6)

            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_bbox_targets,
                weight=weight_targets,
                avg_factor=weight_denorm)
            loss_angle = self.loss_angle(
                pos_angle_preds,
                self.angle_coder.encode(pos_angle_targets).squeeze(-1),
                weight=weight_targets,
                avg_factor=weight_denorm)
            # if self.separate_angle:
            # loss_angle = self.loss_angle(
            #     pos_angle_preds,
            #     pos_angle_targets,
            #     weight=weight_targets,
            #     avg_factor=weight_denorm)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_angle = pos_angle_preds.sum()
            # if self.separate_angle:
            #     loss_angle = pos_angle_preds.sum()

        loss_cls = self.loss_cls(
            flatten_cls_scores, (flatten_labels, score), avg_factor=num_pos)

        # if self.separate_angle:
        return dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_angle=loss_angle)
        # else:
        #     return dict(
        #         loss_cls=loss_cls,
        #         loss_bbox=loss_bbox)

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                angle_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (
                cls_score, bbox_pred, angle_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, angle_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            # dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                # -1, self.angle_coder.encode_size)
                -1,
                self.reg_max + 1)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(
                    bbox_pred=bbox_pred, angle_pred=angle_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            angle_pred = filtered_results['angle_pred']
            priors = filtered_results['priors']

            decoded_angle = self.integral(angle_pred)
            decoded_angle = self.angle_coder.decode(decoded_angle)

            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

    def filter_bboxes(self, cls_scores: List[Tensor], bbox_preds: List[Tensor],
                      angle_preds: List[Tensor]) -> List[List[Tensor]]:
        """This function will be used in S2ANet, whose num_anchors=1.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 5, H, W)

        Returns:
            list[list[Tensor]]: refined rbboxes of each level of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds) == len(angle_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            if self.norm_on_bbox and self.training:
                bbox_pred = bbox_preds[lvl] * self.strides[lvl]
            else:
                bbox_pred = bbox_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 4)

            angle_pred = angle_preds[lvl]
            angle_pred = angle_pred.permute(0, 2, 3, 1)

            angle_pred = self.integral(angle_pred)
            angle_pred = self.angle_coder.decode(angle_pred)

            angle_pred = angle_pred.reshape(num_imgs, -1, 1)

            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=-1)

            priors = mlvl_priors[lvl]

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(priors, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list


# @MODELS.register_module()
# class WS2ARefineDIHead(WS2ARefineHead):
#
#     def _init_layers(self) -> None:
#         """Initialize layers of the head."""
#         self.or_conv_reg = ORConv2d(
#             self.feat_channels,
#             int(self.feat_channels / 8),
#             kernel_size=3,
#             padding=1,
#             arf_config=(1, 8))
#         self.or_conv_cls = ORConv2d(
#             self.feat_channels,
#             int(self.feat_channels / 8),
#             kernel_size=3,
#             padding=1,
#             arf_config=(1, 8))
#         self.or_pool_cls = RotationInvariantPooling(256, 8)
#         self.relu = nn.ReLU(inplace=True)
#         self.cls_convs = nn.ModuleList()
#         self.reg_convs = nn.ModuleList()
#         for i in range(self.stacked_convs):
#             chn = int(self.feat_channels / 8) if i == 0 else self.feat_channels
#             self.cls_convs.append(
#                 ConvModule(
#                     chn,
#                     self.feat_channels,
#                     3,
#                     stride=1,
#                     padding=1,
#                     conv_cfg=self.conv_cfg,
#                     norm_cfg=self.norm_cfg))
#             self.reg_convs.append(
#                 ConvModule(
#                     self.feat_channels,
#                     self.feat_channels,
#                     3,
#                     stride=1,
#                     padding=1,
#                     conv_cfg=self.conv_cfg,
#                     norm_cfg=self.norm_cfg))
#         self.retina_cls = nn.Conv2d(
#             self.feat_channels,
#             self.num_base_priors * self.cls_out_channels,
#             3,
#             padding=1)
#         reg_dim = self.bbox_coder.encode_size
#         self.retina_reg = nn.Conv2d(
#             self.feat_channels, self.num_base_priors * reg_dim, 3, padding=1)
#
#     def forward_single(self, x: Tuple[Tensor, Tensor]) -> Tuple[
#         Tensor, Tensor]:
#         """Forward feature of a single scale level.
#
#         Args:
#             x (Tensor): Features of a single scale level.
#
#         Returns:
#             tuple:
#
#             - cls_score (Tensor): Cls scores for a single scale level
#               the channels number is num_anchors * num_classes.
#             - bbox_pred (Tensor): Box energies / deltas for a single scale
#               level, the channels number is num_anchors * 4.
#         """
#         x_cls, x_reg = x
#         cls_feat = self.or_pool_cls(self.or_conv_cls(x_cls))
#         reg_feat = self.or_conv_reg(x_reg)
#         for cls_conv in self.cls_convs:
#             cls_feat = cls_conv(cls_feat)
#         for reg_conv in self.reg_convs:
#             reg_feat = reg_conv(reg_feat)
#         cls_score = self.retina_cls(cls_feat)
#         bbox_pred = self.retina_reg(reg_feat)
#         return cls_score, bbox_pred
#
#     def loss_by_feat(
#             self,
#             cls_scores: List[Tensor],
#             bbox_preds: List[Tensor],
#             batch_gt_instances: InstanceList,
#             batch_img_metas: List[dict],
#             batch_gt_instances_ignore: OptInstanceList = None,
#             rois: List[Tensor] = None) -> dict:
#         assert rois is not None
#         self.bboxes_as_anchors = rois
#
#         featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
#         assert len(featmap_sizes) == self.prior_generator.num_levels
#
#         device = cls_scores[0].device
#         anchor_list, valid_flag_list = self.get_anchors(
#             featmap_sizes, batch_img_metas, device=device)
#
#         cls_reg_targets = self.get_targets(
#             anchor_list,
#             valid_flag_list,
#             batch_gt_instances,
#             batch_img_metas,
#             batch_gt_instances_ignore=batch_gt_instances_ignore)
#
#         (anchor_list, labels_list, label_weights_list, bbox_targets_list,
#          bbox_weights_list, avg_factor) = cls_reg_targets
#         avg_factor = reduce_mean(
#             torch.tensor(avg_factor, dtype=torch.float, device=device)).item()
#
#         losses_cls, losses_bbox = multi_apply(
#             self.loss_by_feat_single,
#             anchor_list,
#             cls_scores,
#             bbox_preds,
#             labels_list,
#             label_weights_list,
#             bbox_targets_list,
#             bbox_weights_list,
#             avg_factor=avg_factor)
#
#         return dict(
#             loss_cls=losses_cls,
#             loss_bbox=losses_bbox)
#
#     def loss_by_feat_single(self,anchors: Tensor, cls_score: Tensor,
#                             bbox_pred: Tensor, labels: Tensor,
#                             label_weights: Tensor, bbox_targets: Tensor,
#                             bbox_weights: Tensor, avg_factor: int) -> tuple:
#         """Calculate the loss of a single scale level based on the features
#         extracted by the detection head.
#
#         Args:
#             cls_score (Tensor): Box scores for each scale level
#                 Has shape (N, num_anchors * num_classes, H, W).
#             bbox_pred (Tensor): Box energies / deltas for each scale
#                 level with shape (N, num_anchors * 4, H, W).
#             anchors (Tensor): Box reference for each scale level with shape
#                 (N, num_total_anchors, 4).
#             labels (Tensor): Labels of each anchors with shape
#                 (N, num_total_anchors).
#             label_weights (Tensor): Label weights of each anchor with shape
#                 (N, num_total_anchors)
#             bbox_targets (Tensor): BBox regression targets of each anchor
#                 weight shape (N, num_total_anchors, 4).
#             bbox_weights (Tensor): BBox regression loss weights of each anchor
#                 with shape (N, num_total_anchors, 4).
#             avg_factor (int): Average factor that is used to average the loss.
#
#         Returns:
#             tuple: loss components.
#         """
#         # classification loss
#         labels = labels.reshape(-1)
#         label_weights = label_weights.reshape(-1)
#         cls_score = cls_score.permute(0, 2, 3,
#                                       1).reshape(-1, self.cls_out_channels)
#
#         bg_class_ind = self.num_classes
#         pos_inds = ((labels >= 0)
#                     & (labels < bg_class_ind)).nonzero().squeeze(1)
#         score = label_weights.new_zeros(labels.shape)
#
#         target_dim = bbox_targets.size(-1)
#         bbox_targets = bbox_targets.reshape(-1, target_dim)
#         bbox_weights = bbox_weights.reshape(-1, target_dim)
#         bbox_pred = bbox_pred.permute(0, 2, 3,
#                                       1).reshape(-1,
#                                                  self.bbox_coder.encode_size)
#
#         if self.reg_decoded_bbox:
#             # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
#             # is applied directly on the decoded bounding boxes, it
#             # decodes the already encoded coordinates to absolute format.
#             anchors = anchors.reshape(-1, anchors.size(-1))
#             bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
#             bbox_pred = get_box_tensor(bbox_pred)
#
#         if len(pos_inds) > 0:
#
#             pos_bbox_pred = bbox_pred[pos_inds]
#             pos_bbox_targets = bbox_targets[pos_inds]
#
#             if not self.reg_decoded_bbox:
#                 anchors = anchors.reshape(-1, anchors.size(-1))
#                 pos_anchors = anchors[pos_inds]
#
#                 pos_bbox_pred = self.bbox_coder.decode(pos_anchors,
#                                                        pos_bbox_pred)
#                 pos_bbox_pred = get_box_tensor(pos_bbox_pred)
#                 pos_bbox_targets = self.bbox_coder.decode(pos_anchors,
#                                                           pos_bbox_targets)
#                 pos_bbox_targets = get_box_tensor(pos_bbox_targets)
#
#             score[pos_inds] = rbbox_overlaps(
#                 pos_bbox_pred.detach(),
#                 pos_bbox_targets,
#                 is_aligned=True)
#
#         loss_bbox = self.loss_bbox(
#             bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
#         # cls (qfl) loss
#         loss_cls = self.loss_cls(
#             cls_score, (labels, score),
#             weight=label_weights,
#             avg_factor=avg_factor)
#
#         return loss_cls, loss_bbox
#
#     def get_targets(self,
#                     anchor_list: List[List[Tensor]],
#                     valid_flag_list: List[List[Tensor]],
#                     batch_gt_instances: InstanceList,
#                     batch_img_metas: List[dict],
#                     batch_gt_instances_ignore: OptInstanceList = None,
#                     unmap_outputs: bool = True) -> tuple:
#         """Get targets for ATSS head.
#
#         This method is almost the same as `AnchorHead.get_targets()`. Besides
#         returning the targets as the parent method does, it also returns the
#         anchors as the first element of the returned tuple.
#         """
#         num_imgs = len(batch_img_metas)
#         assert len(anchor_list) == len(valid_flag_list) == num_imgs
#
#         # anchor number of multi levels
#         num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
#         num_level_anchors_list = [num_level_anchors] * num_imgs
#
#         # concat all level anchors and flags to a single tensor
#         for i in range(num_imgs):
#             assert len(anchor_list[i]) == len(valid_flag_list[i])
#             anchor_list[i] = cat_boxes(anchor_list[i])
#             valid_flag_list[i] = torch.cat(valid_flag_list[i])
#
#         # compute targets for each image
#         if batch_gt_instances_ignore is None:
#             batch_gt_instances_ignore = [None] * num_imgs
#         (all_anchors, all_labels, all_label_weights, all_bbox_targets,
#          all_bbox_weights, pos_inds_list, neg_inds_list,
#          sampling_results_list) = multi_apply(
#             self._get_targets_single,
#             anchor_list,
#             valid_flag_list,
#             num_level_anchors_list,
#             batch_gt_instances,
#             batch_img_metas,
#             batch_gt_instances_ignore,
#             unmap_outputs=unmap_outputs)
#         # Get `avg_factor` of all images, which calculate in `SamplingResult`.
#         # When using sampling method, avg_factor is usually the sum of
#         # positive and negative priors. When using `PseudoSampler`,
#         # `avg_factor` is usually equal to the number of positive priors.
#         avg_factor = sum(
#             [results.avg_factor for results in sampling_results_list])
#         # split targets to a list w.r.t. multiple levels
#         anchors_list = images_to_levels(all_anchors, num_level_anchors)
#         labels_list = images_to_levels(all_labels, num_level_anchors)
#         label_weights_list = images_to_levels(all_label_weights,
#                                               num_level_anchors)
#         bbox_targets_list = images_to_levels(all_bbox_targets,
#                                              num_level_anchors)
#         bbox_weights_list = images_to_levels(all_bbox_weights,
#                                              num_level_anchors)
#         return (anchors_list, labels_list, label_weights_list,
#                 bbox_targets_list, bbox_weights_list, avg_factor)
#
#     def _get_targets_single(self,
#                             flat_anchors: Tensor,
#                             valid_flags: Tensor,
#                             num_level_anchors: List[int],
#                             gt_instances: InstanceData,
#                             img_meta: dict,
#                             gt_instances_ignore: Optional[InstanceData] = None,
#                             unmap_outputs: bool = True) -> tuple:
#         """Compute regression, classification targets for anchors in a single
#         image.
#         Args:
#             flat_anchors (Tensor): Multi-level anchors of the image, which are
#                 concatenated into a single tensor of shape (num_anchors ,4)
#             valid_flags (Tensor): Multi level valid flags of the image,
#                 which are concatenated into a single tensor of
#                     shape (num_anchors,).
#             num_level_anchors (List[int]): Number of anchors of each scale
#                 level.
#             gt_instances (:obj:`InstanceData`): Ground truth of instance
#                 annotations. It usually includes ``bboxes`` and ``labels``
#                 attributes.
#             img_meta (dict): Meta information for current image.
#             gt_instances_ignore (:obj:`InstanceData`, optional): Instances
#                 to be ignored during training. It includes ``bboxes`` attribute
#                 data that is ignored during training and testing.
#                 Defaults to None.
#             unmap_outputs (bool): Whether to map outputs back to the original
#                 set of anchors.
#         Returns:
#             tuple: N is the number of total anchors in the image.
#                 labels (Tensor): Labels of all anchors in the image with shape
#                     (N,).
#                 label_weights (Tensor): Label weights of all anchor in the
#                     image with shape (N,).
#                 bbox_targets (Tensor): BBox targets of all anchors in the
#                     image with shape (N, 4).
#                 bbox_weights (Tensor): BBox weights of all anchors in the
#                     image with shape (N, 4)
#                 pos_inds (Tensor): Indices of positive anchor with shape
#                     (num_pos,).
#                 neg_inds (Tensor): Indices of negative anchor with shape
#                     (num_neg,).
#                 sampling_result (:obj:`SamplingResult`): Sampling results.
#         """
#         inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
#                                            img_meta['img_shape'][:2],
#                                            self.train_cfg['allowed_border'])
#         if not inside_flags.any():
#             raise ValueError(
#                 'There is no valid anchor inside the image boundary. Please '
#                 'check the image size and anchor sizes, or set '
#                 '``allowed_border`` to -1 to skip the condition.')
#         # assign gt and sample anchors
#         anchors = flat_anchors[inside_flags]
#
#         num_level_anchors_inside = self.get_num_level_anchors_inside(
#             num_level_anchors, inside_flags)
#         pred_instances = InstanceData(priors=anchors)
#         assign_result = self.assigner.assign(pred_instances,
#                                              num_level_anchors_inside,
#                                              gt_instances, gt_instances_ignore)
#
#         sampling_result = self.sampler.sample(assign_result, pred_instances,
#                                               gt_instances)
#
#         num_valid_anchors = anchors.shape[0]
#         target_dim = gt_instances.bboxes.size(-1) if self.reg_decoded_bbox \
#             else self.bbox_coder.encode_size
#         bbox_targets = anchors.new_zeros(num_valid_anchors, target_dim)
#         bbox_weights = anchors.new_zeros(num_valid_anchors, target_dim)
#         labels = anchors.new_full((num_valid_anchors, ),
#                                   self.num_classes,
#                                   dtype=torch.long)
#         label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
#
#         pos_inds = sampling_result.pos_inds
#         neg_inds = sampling_result.neg_inds
#         if len(pos_inds) > 0:
#             if self.reg_decoded_bbox:
#                 pos_bbox_targets = sampling_result.pos_gt_bboxes
#                 pos_bbox_targets = get_box_tensor(pos_bbox_targets)
#             else:
#                 pos_bbox_targets = self.bbox_coder.encode(
#                     sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
#
#             bbox_targets[pos_inds] = pos_bbox_targets
#             bbox_weights[pos_inds, :] = 1.0
#
#             labels[pos_inds] = sampling_result.pos_gt_labels
#             if self.train_cfg['pos_weight'] <= 0:
#                 label_weights[pos_inds] = 1.0
#             else:
#                 label_weights[pos_inds] = self.train_cfg['pos_weight']
#         if len(neg_inds) > 0:
#             label_weights[neg_inds] = 1.0
#
#         # map up to original set of anchors
#         if unmap_outputs:
#             num_total_anchors = flat_anchors.size(0)
#             anchors = unmap(anchors.tensor, num_total_anchors, inside_flags)
#             labels = unmap(
#                 labels, num_total_anchors, inside_flags, fill=self.num_classes)
#             label_weights = unmap(label_weights, num_total_anchors,
#                                   inside_flags)
#             bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
#             bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
#
#         return (anchors, labels, label_weights, bbox_targets, bbox_weights,
#                 pos_inds, neg_inds, sampling_result)
#
#
#     def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
#         """Get the number of valid anchors in every level."""
#
#         split_inside_flags = torch.split(inside_flags, num_level_anchors)
#         num_level_anchors_inside = [
#             int(flags.sum()) for flags in split_inside_flags
#         ]
#         return num_level_anchors_inside


@MODELS.register_module()
class WS2ARefineDIHead(WS2ARefineHead):

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.or_conv_reg = ORConv2d(
            self.feat_channels,
            int(self.feat_channels / 8),
            kernel_size=3,
            padding=1,
            arf_config=(1, 8))
        self.or_conv_cls = ORConv2d(
            self.feat_channels,
            int(self.feat_channels / 8),
            kernel_size=3,
            padding=1,
            arf_config=(1, 8))
        self.or_pool_cls = RotationInvariantPooling(256, 8)
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = int(self.feat_channels / 8) if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        reg_dim = self.bbox_coder.encode_size
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * reg_dim, 3, padding=1)

    def forward_single(self, x: Tuple[Tensor,
                                      Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:

            - cls_score (Tensor): Cls scores for a single scale level
              the channels number is num_anchors * num_classes.
            - bbox_pred (Tensor): Box energies / deltas for a single scale
              level, the channels number is num_anchors * 4.
        """
        x_cls, x_reg = x
        cls_feat = self.or_pool_cls(self.or_conv_cls(x_cls))
        reg_feat = self.or_conv_reg(x_reg)
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    # def loss_by_feat(self,
    #                  cls_scores: List[Tensor],
    #                  bbox_preds: List[Tensor],
    #                  batch_gt_instances: InstanceList,
    #                  batch_img_metas: List[dict],
    #                  batch_gt_instances_ignore: OptInstanceList = None,
    #                  rois: List[Tensor] = None) -> dict:
    #     """Calculate the loss based on the features extracted by the detection
    #     head.
    #
    #     Args:
    #         cls_scores (list[Tensor]): Box scores for each scale level
    #             has shape (N, num_anchors * num_classes, H, W).
    #         bbox_preds (list[Tensor]): Box energies / deltas for each scale
    #             level with shape (N, num_anchors * 4, H, W).
    #         batch_gt_instances (list[:obj:`InstanceData`]): Batch of
    #             gt_instance. It usually includes ``bboxes`` and ``labels``
    #             attributes.
    #         batch_img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #         batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
    #             Batch of gt_instances_ignore. It includes ``bboxes`` attribute
    #             data that is ignored during training and testing.
    #             Defaults to None.
    #         rois (list[Tensor])
    #
    #     Returns:
    #         dict: A dictionary of loss components.
    #     """
    #     assert rois is not None
    #     self.bboxes_as_anchors = rois
    #     featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    #     assert len(featmap_sizes) == self.prior_generator.num_levels
    #
    #     device = cls_scores[0].device
    #
    #     anchor_list, valid_flag_list = self.get_anchors(
    #         featmap_sizes, batch_img_metas, device=device)
    #     cls_reg_targets = self.get_targets(
    #         anchor_list,
    #         valid_flag_list,
    #         batch_gt_instances,
    #         batch_img_metas,
    #         batch_gt_instances_ignore=batch_gt_instances_ignore)
    #     (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
    #      avg_factor) = cls_reg_targets
    #
    #     num_imgs = cls_scores[0].size(0)
    #     # flatten cls_scores, bbox_preds, angle_preds and centerness
    #     flatten_cls_scores = [
    #         cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
    #         for cls_score in cls_scores
    #     ]
    #     flatten_bbox_preds = [
    #         bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
    #         for bbox_pred in bbox_preds
    #     ]
    #     flatten_labels = [
    #         label.reshape(-1)
    #         for label in labels_list
    #     ]
    #     flatten_label_weights = [
    #         label_weight.reshape(-1)
    #         for label_weight in label_weights_list
    #     ]
    #     flatten_bbox_targets = [
    #         bbox_targets.reshape(-1, 5)
    #         for bbox_targets in bbox_targets_list
    #     ]
    #
    #     flatten_cls_scores = torch.cat(flatten_cls_scores)
    #     flatten_bbox_preds = torch.cat(flatten_bbox_preds)
    #     flatten_labels = torch.cat(flatten_labels)
    #     flatten_label_weights = torch.cat(flatten_label_weights)
    #     flatten_bbox_targets = torch.cat(flatten_bbox_targets)
    #
    #     # concat all level anchors and flags to a single tensor
    #     concat_anchor_list = []
    #     for i in range(len(anchor_list)):
    #         concat_anchor_list.append(cat_boxes(anchor_list[i]))
    #     flatten_anchors = cat_boxes(concat_anchor_list)
    #
    #
    #     bg_class_ind = self.num_classes
    #     pos_inds = ((flatten_labels >= 0)
    #                 & (flatten_labels < bg_class_ind)).nonzero().squeeze(1)
    #     score = flatten_label_weights.new_zeros(flatten_labels.shape)
    #
    #     if len(pos_inds) > 0:
    #         bbox_targets = flatten_bbox_targets
    #         bbox_pred = flatten_bbox_preds
    #         pos_bbox_targets = bbox_targets[pos_inds]
    #         pos_bbox_pred = bbox_pred[pos_inds]
    #         anchors = flatten_anchors
    #         pos_anchors = anchors[pos_inds]
    #
    #         weight_targets = flatten_cls_scores.detach().sigmoid()
    #         weight_targets = weight_targets.max(dim=1, keepdim=True)[0][
    #             pos_inds]
    #
    #         pos_decoded_bbox_preds = self.bbox_coder.decode(pos_anchors,
    #                                                         pos_bbox_pred)
    #         pos_decoded_bbox_preds = get_box_tensor(pos_decoded_bbox_preds)
    #
    #         pos_decoded_bbox_targets = pos_bbox_targets
    #         if self.reg_decoded_bbox:
    #             pos_bbox_pred = pos_decoded_bbox_preds
    #         else:
    #             pos_decoded_bbox_targets = self.bbox_coder.decode(pos_anchors,
    #                                                               pos_bbox_targets)
    #             pos_decoded_bbox_targets = get_box_tensor(
    #                 pos_decoded_bbox_targets)
    #
    #         score[pos_inds] = rbbox_overlaps(
    #             pos_decoded_bbox_preds.detach(),
    #             pos_decoded_bbox_targets,
    #             is_aligned=True)
    #
    #         # regression loss
    #         loss_bbox = self.loss_bbox(
    #             pos_bbox_pred,
    #             pos_bbox_targets,
    #             weight=weight_targets,
    #             avg_factor=len(pos_inds))
    #     else:
    #         loss_bbox = flatten_bbox_preds.sum() * 0
    #
    #     # cls (qfl) loss
    #     loss_cls = self.loss_cls(
    #         flatten_cls_scores, (flatten_labels, score),
    #         weight=flatten_label_weights,
    #         avg_factor=avg_factor)
    #
    #     return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)
    # #
    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            anchors: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            bbox_weights: Tensor, avg_factor: int) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            tuple: loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1,
                                                 self.bbox_coder.encode_size)

        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)

        if len(pos_inds) > 0:

            pos_bbox_pred = bbox_pred[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]

            if not self.reg_decoded_bbox:
                anchors = anchors.reshape(-1, anchors.size(-1))
                pos_anchors = anchors[pos_inds]

                pos_bbox_pred = self.bbox_coder.decode(pos_anchors,
                                                       pos_bbox_pred)
                pos_bbox_pred = get_box_tensor(pos_bbox_pred)
                pos_bbox_targets = self.bbox_coder.decode(
                    pos_anchors, pos_bbox_targets)
                pos_bbox_targets = get_box_tensor(pos_bbox_targets)

            score[pos_inds] = rbbox_overlaps(
                pos_bbox_pred.detach(), pos_bbox_targets, is_aligned=True)

        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=avg_factor)

        return loss_cls, loss_bbox


@MODELS.register_module()
class WS2ARefineDISAHead(WS2ARefineHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 frm_cfg: dict = None,
                 reg_max: int = 16,
                 loss_angle: ConfigType = dict(
                     type='mmdet.L1Loss', loss_weight=1.0),
                 angle_coder: ConfigType = dict(type='PseudoAngleCoder'),
                 **kwargs) -> None:
        self.angle_coder = TASK_UTILS.build(angle_coder)
        self.reg_max = reg_max
        super().__init__(num_classes, in_channels, frm_cfg, **kwargs)
        self.loss_angle = MODELS.build(loss_angle)
        self.integral = Integral(reg_max)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.or_conv_reg = ORConv2d(
            self.feat_channels,
            int(self.feat_channels / 8),
            kernel_size=3,
            padding=1,
            arf_config=(1, 8))
        self.or_conv_cls = ORConv2d(
            self.feat_channels,
            int(self.feat_channels / 8),
            kernel_size=3,
            padding=1,
            arf_config=(1, 8))
        self.or_pool_cls = RotationInvariantPooling(256, 8)
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = int(self.feat_channels / 8) if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        reg_dim = 4
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * reg_dim, 3, padding=1)
        self.conv_angle = nn.Conv2d(
            # self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
            self.feat_channels,
            self.num_base_priors * (self.reg_max + 1),
            3,
            padding=1)

    def forward_single(
            self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:

            - cls_score (Tensor): Cls scores for a single scale level
              the channels number is num_anchors * num_classes.
            - bbox_pred (Tensor): Box energies / deltas for a single scale
              level, the channels number is num_anchors * 4.
        """
        x_cls, x_reg = x
        cls_feat = self.or_pool_cls(self.or_conv_cls(x_cls))
        reg_feat = self.or_conv_reg(x_reg)
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        angle_pred = self.conv_angle(reg_feat)

        return cls_score, bbox_pred, angle_pred

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     angle_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     rois: List[Tensor] = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            rois (list[Tensor])

        Returns:
            dict: A dictionary of loss components.
        """
        assert rois is not None
        self.bboxes_as_anchors = rois
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor) = cls_reg_targets

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox, loss_angle = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            angle_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor=avg_factor)
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_angle=loss_angle)

    # def loss_by_feat(self,
    #                  cls_scores: List[Tensor],
    #                  bbox_preds: List[Tensor],
    #                  batch_gt_instances: InstanceList,
    #                  batch_img_metas: List[dict],
    #                  batch_gt_instances_ignore: OptInstanceList = None,
    #                  rois: List[Tensor] = None) -> dict:
    #     """Calculate the loss based on the features extracted by the detection
    #     head.
    #
    #     Args:
    #         cls_scores (list[Tensor]): Box scores for each scale level
    #             has shape (N, num_anchors * num_classes, H, W).
    #         bbox_preds (list[Tensor]): Box energies / deltas for each scale
    #             level with shape (N, num_anchors * 4, H, W).
    #         batch_gt_instances (list[:obj:`InstanceData`]): Batch of
    #             gt_instance. It usually includes ``bboxes`` and ``labels``
    #             attributes.
    #         batch_img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #         batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
    #             Batch of gt_instances_ignore. It includes ``bboxes`` attribute
    #             data that is ignored during training and testing.
    #             Defaults to None.
    #         rois (list[Tensor])
    #
    #     Returns:
    #         dict: A dictionary of loss components.
    #     """
    #     assert rois is not None
    #     self.bboxes_as_anchors = rois
    #     featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    #     assert len(featmap_sizes) == self.prior_generator.num_levels
    #
    #     device = cls_scores[0].device
    #
    #     anchor_list, valid_flag_list = self.get_anchors(
    #         featmap_sizes, batch_img_metas, device=device)
    #     cls_reg_targets = self.get_targets(
    #         anchor_list,
    #         valid_flag_list,
    #         batch_gt_instances,
    #         batch_img_metas,
    #         batch_gt_instances_ignore=batch_gt_instances_ignore)
    #     (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
    #      avg_factor) = cls_reg_targets
    #
    #     num_imgs = cls_scores[0].size(0)
    #     # flatten cls_scores, bbox_preds, angle_preds and centerness
    #     flatten_cls_scores = [
    #         cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
    #         for cls_score in cls_scores
    #     ]
    #     flatten_bbox_preds = [
    #         bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
    #         for bbox_pred in bbox_preds
    #     ]
    #     flatten_labels = [
    #         label.reshape(-1)
    #         for label in labels_list
    #     ]
    #     flatten_label_weights = [
    #         label_weight.reshape(-1)
    #         for label_weight in label_weights_list
    #     ]
    #     flatten_bbox_targets = [
    #         bbox_targets.reshape(-1, 5)
    #         for bbox_targets in bbox_targets_list
    #     ]
    #
    #     flatten_cls_scores = torch.cat(flatten_cls_scores)
    #     flatten_bbox_preds = torch.cat(flatten_bbox_preds)
    #     flatten_labels = torch.cat(flatten_labels)
    #     flatten_label_weights = torch.cat(flatten_label_weights)
    #     flatten_bbox_targets = torch.cat(flatten_bbox_targets)
    #
    #     # concat all level anchors and flags to a single tensor
    #     concat_anchor_list = []
    #     for i in range(len(anchor_list)):
    #         concat_anchor_list.append(cat_boxes(anchor_list[i]))
    #     flatten_anchors = cat_boxes(concat_anchor_list)
    #
    #
    #     bg_class_ind = self.num_classes
    #     pos_inds = ((flatten_labels >= 0)
    #                 & (flatten_labels < bg_class_ind)).nonzero().squeeze(1)
    #     score = flatten_label_weights.new_zeros(flatten_labels.shape)
    #
    #     if len(pos_inds) > 0:
    #         bbox_targets = flatten_bbox_targets
    #         bbox_pred = flatten_bbox_preds
    #         pos_bbox_targets = bbox_targets[pos_inds]
    #         pos_bbox_pred = bbox_pred[pos_inds]
    #         anchors = flatten_anchors
    #         pos_anchors = anchors[pos_inds]
    #
    #         weight_targets = flatten_cls_scores.detach().sigmoid()
    #         weight_targets = weight_targets.max(dim=1, keepdim=True)[0][
    #             pos_inds]
    #
    #         pos_decoded_bbox_preds = self.bbox_coder.decode(pos_anchors,
    #                                                         pos_bbox_pred)
    #         pos_decoded_bbox_preds = get_box_tensor(pos_decoded_bbox_preds)
    #
    #         pos_decoded_bbox_targets = pos_bbox_targets
    #         if self.reg_decoded_bbox:
    #             pos_bbox_pred = pos_decoded_bbox_preds
    #         else:
    #             pos_decoded_bbox_targets = self.bbox_coder.decode(pos_anchors,
    #                                                               pos_bbox_targets)
    #             pos_decoded_bbox_targets = get_box_tensor(
    #                 pos_decoded_bbox_targets)
    #
    #         score[pos_inds] = rbbox_overlaps(
    #             pos_decoded_bbox_preds.detach(),
    #             pos_decoded_bbox_targets,
    #             is_aligned=True)
    #
    #         # regression loss
    #         loss_bbox = self.loss_bbox(
    #             pos_bbox_pred,
    #             pos_bbox_targets,
    #             weight=weight_targets,
    #             avg_factor=len(pos_inds))
    #     else:
    #         loss_bbox = flatten_bbox_preds.sum() * 0
    #
    #     # cls (qfl) loss
    #     loss_cls = self.loss_cls(
    #         flatten_cls_scores, (flatten_labels, score),
    #         weight=flatten_label_weights,
    #         avg_factor=avg_factor)
    #
    #     return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)
    # #

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            angle_pred: Tensor, anchors: Tensor,
                            labels: Tensor, label_weights: Tensor,
                            bbox_targets: Tensor, bbox_weights: Tensor,
                            avg_factor: int) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            tuple: loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        angle_pred = angle_pred.permute(0, 2, 3,
                                        1).reshape(-1, self.reg_max + 1)

        decoded_angle_pred = self.integral(angle_pred)
        # decoded_angle_pred = np.pi * decoded_angle_pred / 16 - 0.5 * np.pi
        decoded_angle_pred = self.angle_coder.decode(decoded_angle_pred)

        bbox_pred = torch.cat([bbox_pred, decoded_angle_pred], dim=-1)
        angle_targets = bbox_targets[:, 4]

        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)

        if len(pos_inds) > 0:

            pos_bbox_pred = bbox_pred[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]

            if not self.reg_decoded_bbox:
                anchors = anchors.reshape(-1, anchors.size(-1))
                pos_anchors = anchors[pos_inds]

                pos_bbox_pred = self.bbox_coder.decode(pos_anchors,
                                                       pos_bbox_pred)
                pos_bbox_pred = get_box_tensor(pos_bbox_pred)
                pos_bbox_targets = self.bbox_coder.decode(
                    pos_anchors, pos_bbox_targets)
                pos_bbox_targets = get_box_tensor(pos_bbox_targets)

            score[pos_inds] = rbbox_overlaps(
                pos_bbox_pred.detach(), pos_bbox_targets, is_aligned=True)

            pos_angle_targets = self.angle_coder.encode(
                angle_targets[pos_inds])

            loss_angle = self.loss_angle(angle_pred[pos_inds],
                                         pos_angle_targets)
        else:
            loss_angle = angle_pred[pos_inds].sum()

        # loss_angle = self.loss_angle(
        #     angle_pred,
        #     angle_targets,
        #     weight=angle_weights,
        #     avg_factor=avg_factor
        # )

        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=avg_factor)

        return loss_cls, loss_bbox, loss_angle

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        angle_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        rois: List[Tensor] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            rois (list[Tensor]):
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        assert rois is not None

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            angle_pred_list = select_single_mlvl(
                angle_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                angle_pred_list=angle_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=rois[img_id],
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                angle_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (
                cls_score, bbox_pred, angle_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, angle_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            # dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                # -1, self.angle_coder.encode_size)
                -1,
                self.reg_max + 1)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(
                    bbox_pred=bbox_pred, angle_pred=angle_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            angle_pred = filtered_results['angle_pred']
            priors = filtered_results['priors']

            decoded_angle = self.integral(angle_pred)
            # decoded_angle = np.pi * decoded_angle / 16 - 0.5 * np.pi
            decoded_angle = self.angle_coder.decode(decoded_angle)

            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)


@MODELS.register_module()
class DAWS2ARefineDISAHead(WS2ARefineDISAHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 frm_cfg: dict = None,
                 reg_max: int = 16,
                 loss_angle: ConfigType = dict(
                     type='mmdet.L1Loss', loss_weight=1.0),
                 angle_coder: ConfigType = dict(type='PseudoAngleCoder'),
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, frm_cfg, reg_max,
                         loss_angle, angle_coder, **kwargs)
        if isinstance(self.prior_generator, PseudoRotatedAnchorGenerator):
            self.extra_anchor = False
            self.num_base_priors = 1
        else:
            self.num_base_priors = min(self.num_base_priors - 1, 1)
            self.extra_anchor = True

    def get_anchors(
        self,
        featmap_sizes: List[tuple],
        batch_img_metas: List[dict],
        device: Union[torch.device, str] = 'cuda'
    ) -> Tuple[List[List[Tensor]], List[List[Tensor]]]:

        if not self.extra_anchor:
            return super(DAWS2ARefineDISAHead,
                         self).get_anchors(featmap_sizes, batch_img_metas,
                                           device)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        # anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        # TODO 只能是一个anchor
        shaped_mlvl_anchors = [
            get_box_tensor(anc).t().reshape(5, -1, 1).permute(1, 2, 0)
            for anc in multi_level_anchors
        ]

        anchor_list = [[
            torch.cat(
                [bboxes_img_lvl.unsqueeze(1).detach(), shaped_mlvl_anchors[i]],
                dim=1).view(-1, 5)
            for i, bboxes_img_lvl in enumerate(bboxes_img)
        ] for bboxes_img in self.bboxes_as_anchors]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    # def loss_by_feat(
    #         self,
    #         cls_scores: List[Tensor],
    #         bbox_preds: List[Tensor],
    #         angle_preds: List[Tensor],
    #         batch_gt_instances: InstanceList,
    #         batch_img_metas: List[dict],
    #         batch_gt_instances_ignore: OptInstanceList = None,
    #         rois: List[Tensor] = None) -> dict:
    #     assert rois is not None
    #     self.bboxes_as_anchors = rois
    #
    #     featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    #     assert len(featmap_sizes) == self.prior_generator.num_levels
    #
    #     device = cls_scores[0].device
    #     anchor_list, valid_flag_list = self.get_anchors(
    #         featmap_sizes, batch_img_metas, device=device)
    #
    #     cls_reg_targets = self.get_targets(
    #         anchor_list,
    #         valid_flag_list,
    #         batch_gt_instances,
    #         batch_img_metas,
    #         batch_gt_instances_ignore=batch_gt_instances_ignore)
    #
    #     (anchor_list, labels_list, label_weights_list, bbox_targets_list,
    #      bbox_weights_list, avg_factor) = cls_reg_targets
    #     avg_factor = reduce_mean(
    #         torch.tensor(avg_factor, dtype=torch.float, device=device)).item()
    #     #
    #     # # anchor number of multi levels
    #     # num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    #     # # concat all level anchors and flags to a single tensor
    #     # concat_anchor_list = []
    #     # for i in range(len(anchor_list)):
    #     #     concat_anchor_list.append(cat_boxes(anchor_list[i]))
    #     # all_anchor_list = images_to_levels(concat_anchor_list,
    #     #                                    num_level_anchors)
    #
    #     losses_cls, losses_bbox, loss_angle= multi_apply(
    #         self.loss_by_feat_single,
    #         cls_scores,
    #         bbox_preds,
    #         angle_preds,
    #         anchor_list,
    #         labels_list,
    #         label_weights_list,
    #         bbox_targets_list,
    #         bbox_weights_list,
    #         avg_factor=avg_factor)
    #
    #     return dict(
    #         loss_cls=losses_cls,
    #         loss_bbox=losses_bbox,
    #         loss_angle=loss_angle,
    #     )

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            angle_pred: Tensor, anchors: Tensor,
                            labels: Tensor, label_weights: Tensor,
                            bbox_targets: Tensor, bbox_weights: Tensor,
                            avg_factor: int) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            tuple: loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        angle_pred = angle_pred.permute(0, 2, 3,
                                        1).reshape(-1, self.reg_max + 1)

        decoded_angle_pred = self.integral(angle_pred)
        # decoded_angle_pred = np.pi * decoded_angle_pred / 16 - 0.5 * np.pi
        decoded_angle_pred = self.angle_coder.decode(decoded_angle_pred)

        bbox_pred = torch.cat([bbox_pred, decoded_angle_pred], dim=-1)
        angle_targets = bbox_targets[:, 4]

        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)

        if len(pos_inds) > 0:
            #
            # pos_bbox_pred = bbox_pred[pos_inds]
            # pos_bbox_targets = bbox_targets[pos_inds]

            pos_bbox_pred = bbox_pred[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]

            if not self.reg_decoded_bbox:
                anchors = anchors.reshape(-1, anchors.size(-1))
                pos_anchors = anchors[pos_inds]

                pos_bbox_pred = self.bbox_coder.decode(pos_anchors,
                                                       pos_bbox_pred)
                pos_bbox_pred = get_box_tensor(pos_bbox_pred)
                pos_bbox_targets = self.bbox_coder.decode(
                    pos_anchors, pos_bbox_targets)
                pos_bbox_targets = get_box_tensor(pos_bbox_targets)

            # score[pos_inds] = rbbox_overlaps(
            #     pos_bbox_pred.detach(),
            #     pos_bbox_targets,
            #     is_aligned=True)

            pos_angle_targets = self.angle_coder.encode(
                angle_targets[pos_inds])

            loss_angle = self.loss_angle(angle_pred[pos_inds],
                                         pos_angle_targets)
        else:
            loss_angle = angle_pred[pos_inds].sum()

        # loss_angle = self.loss_angle(
        #     angle_pred,
        #     angle_targets,
        #     weight=angle_weights,
        #     avg_factor=avg_factor
        # )

        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, labels, weight=label_weights, avg_factor=avg_factor)

        return loss_cls, loss_bbox, loss_angle

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        angle_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        rois: List[Tensor] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            rois (list[Tensor]):
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if not self.extra_anchor:
            return super(DAWS2ARefineDISAHead,
                         self).predict_by_feat(cls_scores, bbox_preds,
                                               angle_preds, score_factors,
                                               rois, batch_img_metas, cfg,
                                               rescale, with_nms)

        assert len(cls_scores) == len(bbox_preds)
        assert rois is not None

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)
        shaped_mlvl_anchors = [
            get_box_tensor(anc).t().reshape(5, -1, 1).permute(1, 2, 0)
            for anc in mlvl_priors
        ]

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            angle_pred_list = select_single_mlvl(
                angle_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            roi = rois[img_id]
            prior = [
                torch.cat([roi[lvl].unsqueeze(1), pr], dim=1).view(-1, 5)
                for lvl, pr in enumerate(shaped_mlvl_anchors)
            ]
            # prior = [cat_boxes([rois[img_id][lvl], pr.tensor]) for lvl, pr in enumerate(mlvl_priors)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                angle_pred_list=angle_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=prior,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    # def get_targets(self,
    #                 anchor_list: List[List[Tensor]],
    #                 valid_flag_list: List[List[Tensor]],
    #                 batch_gt_instances: InstanceList,
    #                 batch_img_metas: List[dict],
    #                 batch_gt_instances_ignore: OptInstanceList = None,
    #                 unmap_outputs: bool = True) -> tuple:
    #     """Get targets for ATSS head.
    #
    #     This method is almost the same as `AnchorHead.get_targets()`. Besides
    #     returning the targets as the parent method does, it also returns the
    #     anchors as the first element of the returned tuple.
    #     """
    #     num_imgs = len(batch_img_metas)
    #     assert len(anchor_list) == len(valid_flag_list) == num_imgs
    #
    #     # anchor number of multi levels
    #     num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    #     num_level_anchors_list = [num_level_anchors] * num_imgs
    #
    #     # concat all level anchors and flags to a single tensor
    #     for i in range(num_imgs):
    #         assert len(anchor_list[i]) == len(valid_flag_list[i])
    #         anchor_list[i] = cat_boxes(anchor_list[i])
    #         valid_flag_list[i] = torch.cat(valid_flag_list[i])
    #
    #     # compute targets for each image
    #     if batch_gt_instances_ignore is None:
    #         batch_gt_instances_ignore = [None] * num_imgs
    #     (all_anchors, all_labels, all_label_weights, all_bbox_targets,
    #      all_bbox_weights, pos_inds_list, neg_inds_list,
    #      sampling_results_list) = multi_apply(
    #         self._get_targets_single,
    #         anchor_list,
    #         valid_flag_list,
    #         num_level_anchors_list,
    #         batch_gt_instances,
    #         batch_img_metas,
    #         batch_gt_instances_ignore,
    #         unmap_outputs=unmap_outputs)
    #     # Get `avg_factor` of all images, which calculate in `SamplingResult`.
    #     # When using sampling method, avg_factor is usually the sum of
    #     # positive and negative priors. When using `PseudoSampler`,
    #     # `avg_factor` is usually equal to the number of positive priors.
    #     avg_factor = sum(
    #         [results.avg_factor for results in sampling_results_list])
    #     # split targets to a list w.r.t. multiple levels
    #     anchors_list = images_to_levels(all_anchors, num_level_anchors)
    #     labels_list = images_to_levels(all_labels, num_level_anchors)
    #     label_weights_list = images_to_levels(all_label_weights,
    #                                           num_level_anchors)
    #     bbox_targets_list = images_to_levels(all_bbox_targets,
    #                                          num_level_anchors)
    #     bbox_weights_list = images_to_levels(all_bbox_weights,
    #                                          num_level_anchors)
    #     return (anchors_list, labels_list, label_weights_list,
    #             bbox_targets_list, bbox_weights_list, avg_factor)
    #
    # def _get_targets_single(self,
    #                         flat_anchors: Tensor,
    #                         valid_flags: Tensor,
    #                         num_level_anchors: List[int],
    #                         gt_instances: InstanceData,
    #                         img_meta: dict,
    #                         gt_instances_ignore: Optional[InstanceData] = None,
    #                         unmap_outputs: bool = True) -> tuple:
    #     """Compute regression, classification targets for anchors in a single
    #     image.
    #     Args:
    #         flat_anchors (Tensor): Multi-level anchors of the image, which are
    #             concatenated into a single tensor of shape (num_anchors ,4)
    #         valid_flags (Tensor): Multi level valid flags of the image,
    #             which are concatenated into a single tensor of
    #                 shape (num_anchors,).
    #         num_level_anchors (List[int]): Number of anchors of each scale
    #             level.
    #         gt_instances (:obj:`InstanceData`): Ground truth of instance
    #             annotations. It usually includes ``bboxes`` and ``labels``
    #             attributes.
    #         img_meta (dict): Meta information for current image.
    #         gt_instances_ignore (:obj:`InstanceData`, optional): Instances
    #             to be ignored during training. It includes ``bboxes`` attribute
    #             data that is ignored during training and testing.
    #             Defaults to None.
    #         unmap_outputs (bool): Whether to map outputs back to the original
    #             set of anchors.
    #     Returns:
    #         tuple: N is the number of total anchors in the image.
    #             labels (Tensor): Labels of all anchors in the image with shape
    #                 (N,).
    #             label_weights (Tensor): Label weights of all anchor in the
    #                 image with shape (N,).
    #             bbox_targets (Tensor): BBox targets of all anchors in the
    #                 image with shape (N, 4).
    #             bbox_weights (Tensor): BBox weights of all anchors in the
    #                 image with shape (N, 4)
    #             pos_inds (Tensor): Indices of positive anchor with shape
    #                 (num_pos,).
    #             neg_inds (Tensor): Indices of negative anchor with shape
    #                 (num_neg,).
    #             sampling_result (:obj:`SamplingResult`): Sampling results.
    #     """
    #     inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
    #                                        img_meta['img_shape'][:2],
    #                                        self.train_cfg['allowed_border'])
    #     if not inside_flags.any():
    #         raise ValueError(
    #             'There is no valid anchor inside the image boundary. Please '
    #             'check the image size and anchor sizes, or set '
    #             '``allowed_border`` to -1 to skip the condition.')
    #     # assign gt and sample anchors
    #     anchors = flat_anchors[inside_flags]
    #
    #     num_level_anchors_inside = self.get_num_level_anchors_inside(
    #         num_level_anchors, inside_flags)
    #     pred_instances = InstanceData(priors=anchors)
    #     assign_result = self.assigner.assign(pred_instances,
    #                                          num_level_anchors_inside,
    #                                          gt_instances, gt_instances_ignore)
    #
    #     sampling_result = self.sampler.sample(assign_result, pred_instances,
    #                                           gt_instances)
    #
    #     num_valid_anchors = anchors.shape[0]
    #     target_dim = gt_instances.bboxes.size(-1) if self.reg_decoded_bbox \
    #         else self.bbox_coder.encode_size
    #     bbox_targets = anchors.new_zeros(num_valid_anchors, target_dim)
    #     bbox_weights = anchors.new_zeros(num_valid_anchors, target_dim)
    #     labels = anchors.new_full((num_valid_anchors,),
    #                               self.num_classes,
    #                               dtype=torch.long)
    #     label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
    #
    #     pos_inds = sampling_result.pos_inds
    #     neg_inds = sampling_result.neg_inds
    #     if len(pos_inds) > 0:
    #         if self.reg_decoded_bbox:
    #             pos_bbox_targets = sampling_result.pos_gt_bboxes
    #             pos_bbox_targets = get_box_tensor(pos_bbox_targets)
    #         else:
    #             pos_bbox_targets = self.bbox_coder.encode(
    #                 sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
    #
    #         bbox_targets[pos_inds] = pos_bbox_targets
    #         bbox_weights[pos_inds, :] = 1.0
    #
    #         labels[pos_inds] = sampling_result.pos_gt_labels
    #         if self.train_cfg['pos_weight'] <= 0:
    #             label_weights[pos_inds] = 1.0
    #         else:
    #             label_weights[pos_inds] = self.train_cfg['pos_weight']
    #     if len(neg_inds) > 0:
    #         label_weights[neg_inds] = 1.0
    #
    #     # map up to original set of anchors
    #     if unmap_outputs:
    #         num_total_anchors = flat_anchors.size(0)
    #         anchors = unmap(anchors.tensor, num_total_anchors, inside_flags)
    #         labels = unmap(
    #             labels, num_total_anchors, inside_flags, fill=self.num_classes)
    #         label_weights = unmap(label_weights, num_total_anchors,
    #                               inside_flags)
    #         bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
    #         bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
    #
    #     return (anchors, labels, label_weights, bbox_targets, bbox_weights,
    #             pos_inds, neg_inds, sampling_result)
    #
    # def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
    #     """Get the number of valid anchors in every level."""
    #
    #     split_inside_flags = torch.split(inside_flags, num_level_anchors)
    #     num_level_anchors_inside = [
    #         int(flags.sum()) for flags in split_inside_flags
    #     ]
    #     return num_level_anchors_inside


@MODELS.register_module()
class MAWS2ARefineHead(WS2ARefineDIHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 frm_cfg: dict = None,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, frm_cfg, **kwargs)
        if isinstance(self.prior_generator, PseudoRotatedAnchorGenerator):
            self.extra_anchor = False
            self.num_base_priors = 1
        else:
            self.num_base_priors = min(self.num_base_priors - 1, 1)
            self.extra_anchor = True

    def get_anchors(
        self,
        featmap_sizes: List[tuple],
        batch_img_metas: List[dict],
        device: Union[torch.device, str] = 'cuda'
    ) -> Tuple[List[List[Tensor]], List[List[Tensor]]]:

        if not self.extra_anchor:
            return super(MAWS2ARefineHead,
                         self).get_anchors(featmap_sizes, batch_img_metas,
                                           device)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        # anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        # TODO 只能是一个anchor
        shaped_mlvl_anchors = [
            get_box_tensor(anc).t().reshape(5, -1, 1).permute(1, 2, 0)
            for anc in multi_level_anchors
        ]

        anchor_list = [[
            torch.cat(
                [bboxes_img_lvl.unsqueeze(1).detach(), shaped_mlvl_anchors[i]],
                dim=1).view(-1, 5)
            for i, bboxes_img_lvl in enumerate(bboxes_img)
        ] for bboxes_img in self.bboxes_as_anchors]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            anchors: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            bbox_weights: Tensor, avg_factor: int) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            tuple: loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)

        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1,
                                                 self.bbox_coder.encode_size)

        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)

        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, labels, weight=label_weights, avg_factor=avg_factor)

        return loss_cls, loss_bbox

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        rois: List[Tensor] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            rois (list[Tensor]):
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if not self.extra_anchor:
            return super().predict_by_feat(cls_scores, bbox_preds,
                                           score_factors, rois,
                                           batch_img_metas, cfg, rescale,
                                           with_nms)

        assert len(cls_scores) == len(bbox_preds)
        assert rois is not None

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)
        shaped_mlvl_anchors = [
            get_box_tensor(anc).t().reshape(5, -1, 1).permute(1, 2, 0)
            for anc in mlvl_priors
        ]

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            roi = rois[img_id]
            prior = [
                torch.cat([roi[lvl].unsqueeze(1), pr], dim=1).view(-1, 5)
                for lvl, pr in enumerate(shaped_mlvl_anchors)
            ]
            # prior = [cat_boxes([rois[img_id][lvl], pr.tensor]) for lvl, pr in enumerate(mlvl_priors)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=prior,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list


@MODELS.register_module()
class MAWS2ASepBNRefineHead(MAWS2ARefineHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 share_conv=True,
                 num_ins=5,
                 frm_cfg: dict = None,
                 **kwargs) -> None:
        self.share_conv = share_conv,
        self.num_ins = num_ins
        super().__init__(num_classes, in_channels, frm_cfg, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.or_conv_reg = ORConv2d(
            self.in_channels,
            int(self.feat_channels / 8),
            kernel_size=3,
            padding=1,
            arf_config=(1, 8))
        self.or_conv_cls = ORConv2d(
            self.in_channels,
            int(self.feat_channels / 8),
            kernel_size=3,
            padding=1,
            arf_config=(1, 8))
        self.or_pool_cls = RotationInvariantPooling(self.feat_channels, 8)

        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for n in range(self.num_ins):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = int(self.feat_channels /
                          8) if i == 0 else self.feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
                reg_convs.append(
                    ConvModule(
                        self.feat_channels,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        if self.share_conv:
            for n in range(5):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        reg_dim = self.bbox_coder.encode_size
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * reg_dim, 3, padding=1)

    def forward(self,
                feats: Tuple[Tensor],
                with_feats=False) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        cls_scores = []
        bbox_preds = []
        cls_feats = []
        reg_feats = []
        for i, x in enumerate(feats):
            x_cls, x_reg = x
            cls_feat = self.or_pool_cls(self.or_conv_cls(x_cls))
            reg_feat = self.or_conv_reg(x_reg)

            # cls_feat = feats[i]
            # reg_feat = feats[i]
            for cls_conv in self.cls_convs[i]:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs[i]:
                reg_feat = reg_conv(reg_feat)

            cls_feats.append(cls_feat)
            reg_feats.append(reg_feat)

            cls_score = self.retina_cls(cls_feat)
            bbox_pred = self.retina_reg(reg_feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        if with_feats:
            return cls_scores, bbox_preds, cls_feats, reg_feats
        else:
            return cls_scores, bbox_preds

    def refine_bboxes(self, cls_scores: List[Tensor], bbox_preds: List[Tensor],
                      rois: List[List[Tensor]]) -> List[List[Tensor]]:
        """Refine predicted bounding boxes at each position of the feature
        maps. This method will be used in R3Det in refinement stages.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 5, H, W)
            rois (list[list[Tensor]]): input rbboxes of each level of each
                image. rois output by former stages and are to be refined

        Returns:
            list[list[Tensor]]: best or refined rbboxes of each level of each
            image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        bboxes_list = [[] for _ in range(num_imgs)]

        assert rois is not None
        mlvl_rois = [torch.cat(r) for r in zip(*rois)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            rois = mlvl_rois[lvl]
            assert bbox_pred.size(1) == 5
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(-1, 5)
            refined_bbox = self.bbox_coder.decode(rois, bbox_pred)
            refined_bbox = refined_bbox.reshape(num_imgs, -1, 5)
            for img_id in range(num_imgs):
                bboxes_list[img_id].append(
                    get_box_tensor(refined_bbox[img_id].detach()))
        return bboxes_list

    #     return multi_apply(self.forward_single, x, list(range(5)))
    #
    # def forward_single(self, x: Tuple[Tensor, Tensor], idx: int) -> Tuple[
    #     Tensor, Tensor]:
    #     """Forward feature of a single scale level.
    #
    #     Args:
    #         x (Tensor): Features of a single scale level.
    #
    #     Returns:
    #         tuple:
    #
    #         - cls_score (Tensor): Cls scores for a single scale level
    #           the channels number is num_anchors * num_classes.
    #         - bbox_pred (Tensor): Box energies / deltas for a single scale
    #           level, the channels number is num_anchors * 4.
    #     """
    #     x_cls, x_reg = x
    #     cls_feat = self.or_pool_cls(self.or_conv_cls(x_cls))
    #     reg_feat = self.or_conv_reg(x_reg)
    #     for cls_conv in self.cls_convs:
    #         cls_feat = cls_conv(cls_feat)
    #     for reg_conv in self.reg_convs:
    #         reg_feat = reg_conv(reg_feat)
    #     cls_score = self.retina_cls(cls_feat)
    #     bbox_pred = self.retina_reg(reg_feat)
    #     return cls_score, bbox_pred


@MODELS.register_module()
class QQQIIIHead(QQQHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 regress_ranges: RangeType = ((-1, 64), (64, 128), (128, 256),
                                              (256, 512), (512, INF)),
                 center_sampling: bool = False,
                 center_sample_radius: float = 1.5,
                 norm_on_bbox: bool = False,
                 loss_cls: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(type='IoULoss', loss_weight=1.0),
                 angle_version: str = 'le90',
                 separate_angle: bool = False,
                 scale_angle: bool = True,
                 angle_coder: ConfigType = dict(type='PseudoAngleCoder'),
                 h_bbox_coder: ConfigType = dict(
                     type='mmdet.DistancePointBBoxCoder'),
                 reg_max: int = 16,
                 loss_angle: ConfigType = dict(
                     type='mmdet.L1Loss', loss_weight=1.0),
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, regress_ranges,
                         center_sampling, center_sample_radius, norm_on_bbox,
                         loss_cls, loss_bbox, angle_version, separate_angle,
                         scale_angle, angle_coder, h_bbox_coder, reg_max,
                         loss_angle, norm_cfg, init_cfg, **kwargs)
        self.integral = IIII(self.reg_max)

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        angle_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            angle_preds (list[Tensor]): Box angle for each scale level, each \
                is a 4D-tensor, the channel number is num_points * encode_size.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets, angle_targets = self.get_targets(
            all_level_points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds, angle_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        angle_dim = self.angle_coder.encode_size
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, self.reg_max + 1)
            # angle_pred.permute(0, 2, 3, 1).reshape(-1, angle_dim)
            for angle_pred in angle_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        # pos_angle_targets = self.angle_coder.encode(pos_angle_targets)
        score = pos_bbox_targets.new_zeros(flatten_labels.shape)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            # if self.separate_angle:
            #     bbox_coder = self.h_bbox_coder
            #     overlap_func = bbox_overlaps
            # else:
            bbox_coder = self.bbox_coder
            # pos_angle_preds = self.angle_coder.decode(pos_angle_preds)
            poe_list = self.integral(pos_angle_preds)
            # pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
            # poe = np.pi * poe / 16 - 0.5 * np.pi
            weight_targets = flatten_cls_scores.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            weight_denorm = max(
                reduce_mean(weight_targets.sum().detach()), 1e-6)
            overlap_func = rbbox_overlaps
            pos_bbox_targets = torch.cat([pos_bbox_targets, pos_angle_targets],
                                         dim=-1)
            pos_decoded_bbox_targets = bbox_coder.decode(
                pos_points, pos_bbox_targets)

            loss_bbox = None
            loss_angle = None
            for poe_i in poe_list:
                poe = self.angle_coder.decode(poe_i)

                pos_bbox_preds = torch.cat([pos_bbox_preds, poe], dim=-1)

                pos_decoded_bbox_preds = bbox_coder.decode(
                    pos_points, pos_bbox_preds)

                # score[pos_inds] = overlap_func(
                #     pos_decoded_bbox_preds.detach(),
                #     pos_decoded_bbox_targets,
                #     is_aligned=True)

                loss_bbox_t = self.loss_bbox(
                    pos_decoded_bbox_preds,
                    pos_decoded_bbox_targets,
                    weight=weight_targets,
                    avg_factor=weight_denorm)
                loss_angle_t = self.loss_angle(
                    pos_angle_preds,
                    self.angle_coder.encode(pos_angle_targets).squeeze(-1),
                    weight=weight_targets,
                    avg_factor=weight_denorm)
                if loss_bbox is None:
                    loss_bbox = loss_bbox_t
                    loss_angle = loss_angle_t
                    score[pos_inds] = overlap_func(
                        pos_decoded_bbox_preds.detach(),
                        pos_decoded_bbox_targets,
                        is_aligned=True)

                else:
                    if loss_bbox_t < loss_bbox:
                        loss_bbox = loss_bbox_t
                        loss_angle = loss_angle_t
                        score[pos_inds] = overlap_func(
                            pos_decoded_bbox_preds.detach(),
                            pos_decoded_bbox_targets,
                            is_aligned=True)

            # if self.separate_angle:
            # loss_angle = self.loss_angle(
            #     pos_angle_preds,
            #     pos_angle_targets,
            #     weight=weight_targets,
            #     avg_factor=weight_denorm)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_angle = pos_angle_preds.sum()
            # if self.separate_angle:
            #     loss_angle = pos_angle_preds.sum()

        loss_cls = self.loss_cls(
            flatten_cls_scores, (flatten_labels, score), avg_factor=num_pos)

        # if self.separate_angle:
        return dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_angle=loss_angle)

    def get_targets(
        self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Compute regression, classification and centerness targets for points
        in multiple images.
        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
        Returns:
            tuple: Targets of each level.
            - concat_lvl_labels (list[Tensor]): Labels of each level.
            - concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
            level.
            - concat_lvl_angle_targets (list[Tensor]): Angle targets of \
            each level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, angle_targets_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        angle_targets_list = [
            angle_targets.split(num_points, 0)
            for angle_targets in angle_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_angle_targets)

    def _get_targets_single(
            self, gt_instances: InstanceData, points: Tensor,
            regress_ranges: Tensor,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1))

        areas = gt_bboxes.areas
        gt_bboxes = gt_bboxes.regularize_boxes(self.angle_version)

        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                    inside_gt_bbox_mask)

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        angle_targets = gt_angle[range(num_points), min_area_inds]

        return labels, bbox_targets, angle_targets

    # def loss_by_feat(
    #         self,
    #         cls_scores: List[Tensor],
    #         bbox_preds: List[Tensor],
    #         angle_preds: List[Tensor],
    #         batch_gt_instances: InstanceList,
    #         batch_img_metas: List[dict],
    #         batch_gt_instances_ignore: OptInstanceList = None
    # ) -> Dict[str, Tensor]:
    #     """Calculate the loss based on the features extracted by the detection
    #     head.
    #
    #     Args:
    #         cls_scores (list[Tensor]): Box scores for each scale level,
    #             each is a 4D-tensor, the channel number is
    #             num_points * num_classes.
    #         bbox_preds (list[Tensor]): Box energies / deltas for each scale
    #             level, each is a 4D-tensor, the channel number is
    #             num_points * 4.
    #         angle_preds (list[Tensor]): Box angle for each scale level, each \
    #             is a 4D-tensor, the channel number is num_points * encode_size.
    #         batch_gt_instances (list[:obj:`InstanceData`]): Batch of
    #             gt_instance.  It usually includes ``bboxes`` and ``labels``
    #             attributes.
    #         batch_img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #         batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
    #             Batch of gt_instances_ignore. It includes ``bboxes`` attribute
    #             data that is ignored during training and testing.
    #             Defaults to None.
    #
    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     assert len(cls_scores) == len(bbox_preds) == len(angle_preds)
    #     featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    #     mlvl_priors = self.prior_generator.grid_priors(
    #         featmap_sizes,
    #         dtype=bbox_preds[0].dtype,
    #         device=bbox_preds[0].device,
    #         with_stride=True
    #     )
    #
    #     num_imgs = cls_scores[0].size(0)
    #     # flatten cls_scores, bbox_preds, angle_preds and centerness
    #     flatten_cls_scores = [
    #         cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
    #                                               self.cls_out_channels)
    #         for cls_score in cls_scores
    #     ]
    #     flatten_bbox_preds = [
    #         bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
    #         for bbox_pred in bbox_preds
    #     ]
    #     angle_dim = self.angle_coder.encode_size
    #     flatten_angle_preds = [
    #         angle_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
    #                                                self.reg_max + 1)
    #         # angle_pred.permute(0, 2, 3, 1).reshape(-1, angle_dim)
    #         for angle_pred in angle_preds
    #     ]
    #     flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)
    #     flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    #     flatten_angle_preds = torch.cat(flatten_angle_preds, dim=1)
    #     flatten_points = torch.cat(
    #         [points[:, 0:2].repeat(num_imgs, 1) for points in mlvl_priors])
    #     flatten_priors = torch.cat(mlvl_priors)
    #
    #     (pos_masks, pos_ious, cls_targets, bbox_targets,
    #      num_fg_imgs) = multi_apply(
    #         self._get_targets_single,
    #         flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
    #         flatten_cls_scores.detach(),
    #         flatten_bbox_preds.detach(),
    #         flatten_angle_preds.detach(),
    #         batch_gt_instances, batch_gt_instances_ignore)
    #
    #     num_pos = torch.tensor(
    #         sum(num_fg_imgs),
    #         dtype=torch.float,
    #         device=flatten_cls_scores.device)
    #     num_total_samples = max(reduce_mean(num_pos), 1.0)
    #
    #     pos_masks = torch.cat(pos_masks, 0)
    #     pos_ious = torch.cat(pos_ious, 0)
    #     cls_targets = torch.cat(cls_targets, 0)
    #     bbox_targets = get_box_tensor(cat_boxes(bbox_targets, 0))
    #     angle_targets = bbox_targets[..., 4:5]
    #     score = bbox_targets.new_zeros(cls_targets.shape)
    #     score[pos_masks] = pos_ious
    #
    #     loss_cls = self.loss_cls(
    #         flatten_cls_scores.view(-1, self.num_classes),
    #         (cls_targets, score))/ num_total_samples
    #
    #     if num_pos > 0:
    #         # loss_cls = self.loss_cls(
    #         #     flatten_cls_scores.view(-1, self.num_classes)[pos_masks],
    #         #     cls_targets) / num_total_samples
    #         weight_targets = flatten_cls_scores.view(
    #             -1, self.num_classes)[pos_masks].detach().sigmoid()
    #         weight_targets = weight_targets.max(dim=1)[0]
    #         weight_denorm = max(
    #             reduce_mean(weight_targets.sum().detach()), 1e-6)
    #
    #         pos_bbox_preds = flatten_bbox_preds.view(-1, 4)[pos_masks]
    #         pos_angle_preds = flatten_angle_preds.view(-1, self.reg_max + 1)[
    #             pos_masks]
    #         poe = self.integral(pos_angle_preds)
    #         # pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
    #         # poe = np.pi * poe / 16 - 0.5 * np.pi
    #         poe = self.angle_coder.decode(poe)
    #
    #         pos_bbox_preds = torch.cat([pos_bbox_preds, poe], dim=-1)
    #         pos_bbox_preds = self.bbox_coder.decode(
    #             flatten_points[pos_masks], pos_bbox_preds)
    #
    #         loss_bbox = self.loss_bbox(
    #             pos_bbox_preds,
    #             bbox_targets)/ num_total_samples
    #         loss_angle = self.loss_angle(
    #             pos_angle_preds,
    #             self.angle_coder.encode(angle_targets).squeeze(-1)
    #         )/ num_total_samples
    #
    #     else:
    #         # Avoid cls and reg branch not participating in the gradient
    #         # propagation when there is no ground-truth in the images.
    #         # For more details, please refer to
    #         # https://github.com/open-mmlab/mmdetection/issues/7298
    #         loss_cls = flatten_cls_scores.sum() * 0
    #         loss_bbox = flatten_bbox_preds.sum() * 0
    #         loss_angle = flatten_angle_preds.sum() * 0
    #
    #     return dict(
    #         loss_cls=loss_cls,
    #         loss_bbox=loss_bbox,
    #         loss_angle=loss_angle)
    #     # loss_dict = dict(
    #     #     loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)
    #     #
    #     # flatten_labels = torch.cat(labels)
    #     # flatten_bbox_targets = torch.cat(bbox_targets)
    #     # flatten_angle_targets = torch.cat(angle_targets)
    #     # # repeat points to align with bbox_preds
    #     #
    #     # # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    #     # bg_class_ind = self.num_classes
    #     # pos_inds = ((flatten_labels >= 0)
    #     #             & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
    #     # num_pos = torch.tensor(
    #     #     len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
    #     # num_pos = max(reduce_mean(num_pos), 1.0)
    #     #
    #     # pos_bbox_preds = flatten_bbox_preds[pos_inds]
    #     # pos_angle_preds = flatten_angle_preds[pos_inds]
    #     # pos_bbox_targets = flatten_bbox_targets[pos_inds]
    #     # pos_angle_targets = flatten_angle_targets[pos_inds]
    #     # # pos_angle_targets = self.angle_coder.encode(pos_angle_targets)
    #     # score = pos_bbox_targets.new_zeros(flatten_labels.shape)
    #     #
    #     # if len(pos_inds) > 0:
    #     #     pos_points = flatten_points[pos_inds]
    #     #     if self.separate_angle:
    #     #         bbox_coder = self.h_bbox_coder
    #     #         overlap_func = bbox_overlaps
    #     #     else:
    #     #         bbox_coder = self.bbox_coder
    #     #         # pos_angle_preds = self.angle_coder.decode(pos_angle_preds)
    #     #         poe = self.integral(pos_angle_preds)
    #     #         # pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
    #     #         # poe = np.pi * poe / 16 - 0.5 * np.pi
    #     #         poe = self.angle_coder.decode(poe)
    #     #
    #     #         pos_bbox_preds = torch.cat([pos_bbox_preds, poe],
    #     #                                    dim=-1)
    #     #         pos_bbox_targets = torch.cat(
    #     #             [pos_bbox_targets, pos_angle_targets], dim=-1)
    #     #         overlap_func = rbbox_overlaps
    #     #     pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
    #     #                                                pos_bbox_preds)
    #     #     pos_decoded_bbox_targets = bbox_coder.decode(
    #     #         pos_points, pos_bbox_targets)
    #     #
    #     #     weight_targets = flatten_cls_scores.detach().sigmoid()
    #     #     weight_targets = weight_targets.max(dim=1)[0][pos_inds]
    #     #
    #     #     score[pos_inds] = overlap_func(
    #     #         pos_decoded_bbox_preds.detach(),
    #     #         pos_decoded_bbox_targets,
    #     #         is_aligned=True)
    #     #     weight_denorm = max(
    #     #         reduce_mean(weight_targets.sum().detach()), 1e-6)
    #     #
    #     #     loss_bbox = self.loss_bbox(
    #     #         pos_decoded_bbox_preds,
    #     #         pos_decoded_bbox_targets,
    #     #         weight=weight_targets,
    #     #         avg_factor=weight_denorm)
    #     #     loss_angle = self.loss_angle(
    #     #         pos_angle_preds,
    #     #         self.angle_coder.encode(pos_angle_targets).squeeze(-1),
    #     #         weight=weight_targets,
    #     #         avg_factor=weight_denorm)
    #     #     # if self.separate_angle:
    #     #     # loss_angle = self.loss_angle(
    #     #     #     pos_angle_preds,
    #     #     #     pos_angle_targets,
    #     #     #     weight=weight_targets,
    #     #     #     avg_factor=weight_denorm)
    #     # else:
    #     #     loss_bbox = pos_bbox_preds.sum()
    #     #     loss_angle = pos_angle_preds.sum()
    #     #     # if self.separate_angle:
    #     #     #     loss_angle = pos_angle_preds.sum()
    #     #
    #     # loss_cls = self.loss_cls(
    #     #     flatten_cls_scores, (flatten_labels, score), avg_factor=num_pos)
    #     #
    #     # # if self.separate_angle:
    #     # return dict(
    #     #     loss_cls=loss_cls,
    #     #     loss_bbox=loss_bbox,
    #     #     loss_angle=loss_angle)
    #     # else:
    #     #     return dict(
    #     #         loss_cls=loss_cls,
    #     #         loss_bbox=loss_bbox)
    #
    # @torch.no_grad()
    # def _get_targets_single(
    #         self,
    #         priors: Tensor,
    #         cls_preds: Tensor,
    #         bboxes_preds: Tensor,
    #         angle_preds: Tensor,
    #         gt_instances: InstanceData,
    #         gt_instances_ignore: Optional[InstanceData] = None) -> tuple:
    #     """Compute classification, regression, and objectness targets for
    #     priors in a single image.
    #
    #     Args:
    #         priors (Tensor): All priors of one image, a 2D-Tensor with shape
    #             [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
    #         cls_preds (Tensor): Classification predictions of one image,
    #             a 2D-Tensor with shape [num_priors, num_classes]
    #         decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
    #             a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
    #             br_x, br_y] format.
    #         objectness (Tensor): Objectness predictions of one image,
    #             a 1D-Tensor with shape [num_priors]
    #         gt_instances (:obj:`InstanceData`): Ground truth of instance
    #             annotations. It should includes ``bboxes`` and ``labels``
    #             attributes.
    #         img_meta (dict): Meta information for current image.
    #         gt_instances_ignore (:obj:`InstanceData`, optional): Instances
    #             to be ignored during training. It includes ``bboxes`` attribute
    #             data that is ignored during training and testing.
    #             Defaults to None.
    #     Returns:
    #         tuple:
    #             foreground_mask (list[Tensor]): Binary mask of foreground
    #             targets.
    #             cls_target (list[Tensor]): Classification targets of an image.
    #             obj_target (list[Tensor]): Objectness targets of an image.
    #             bbox_target (list[Tensor]): BBox targets of an image.
    #             l1_target (int): BBox L1 targets of an image.
    #             num_pos_per_img (int): Number of positive samples in an image.
    #     """
    #
    #     num_priors = priors.size(0)
    #     num_gts = len(gt_instances)
    #
    #     gt_instances.bboxes = RotatedBoxes(
    #         gt_instances.bboxes.regularize_boxes(self.angle_version))
    #     # No target
    #     if num_gts == 0:
    #         pos_ious = cls_preds.new_zeros(0)
    #
    #         label_size = cls_preds.size(0)
    #         cls_target = torch.ones(label_size,
    #                                 device=cls_preds.device,
    #                                 dtype=torch.long)
    #         bbox_target = cls_preds.new_zeros((0, 5))
    #         foreground_mask = torch.zeros_like(cls_target).to(torch.bool)
    #
    #         return (foreground_mask, pos_ious, cls_target, bbox_target, 0)
    #
    #     poe = self.integral(angle_preds)
    #     # pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
    #     # poe = np.pi * poe / 16 - 0.5 * np.pi
    #     poe = self.angle_coder.decode(poe)
    #
    #     if self.norm_on_bbox:
    #         bboxes_preds[:, 0::2] = bboxes_preds[:, 0::2] * priors[:, 2:3]
    #         bboxes_preds[:, 1::2] = bboxes_preds[:, 1::2] * priors[:, 3:4]
    #     bbox_preds = torch.cat([bboxes_preds, poe], dim=-1)
    #     decoded_bboxes = self.bbox_coder.decode(priors[:, :2], bbox_preds)
    #
    #     # YOLOX uses center priors with 0.5 offset to assign targets,
    #     # but use center priors without offset to regress bboxes.
    #     offset_priors = torch.cat(
    #         [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)
    #
    #     scores = cls_preds.sigmoid()
    #     pred_instances = InstanceData(
    #         bboxes=decoded_bboxes,
    #         scores=scores.sqrt_(),
    #         priors=offset_priors)
    #     assign_result = self.assigner.assign(
    #         pred_instances=pred_instances,
    #         gt_instances=gt_instances,
    #         gt_instances_ignore=gt_instances_ignore)
    #
    #     sampling_result = self.sampler.sample(assign_result, pred_instances,
    #                                           gt_instances)
    #     pos_inds = sampling_result.pos_inds
    #     num_pos_per_img = pos_inds.size(0)
    #
    #     pos_ious = assign_result.max_overlaps[pos_inds]
    #
    #     label_size = cls_preds.size(0)
    #     cls_target = torch.ones(label_size,
    #                             device=cls_preds.device,
    #                             dtype=torch.long) * self.num_classes
    #     cls_target[pos_inds] = sampling_result.pos_gt_labels
    #     # cls_target = F.one_hot(sampling_result.pos_gt_labels, self.num_classes)
    #
    #     bbox_target = sampling_result.pos_gt_bboxes
    #     bbox_target = get_box_tensor(bbox_target)
    #     # bbox_target = bbox_target.regularize_boxes(self.angle_version)
    #     if self.norm_on_bbox:
    #         coded_box = self.bbox_coder.encode(priors[pos_inds, :2], bbox_target)
    #         coded_box[:, 2:4] = coded_box[:, 2:4] / priors[pos_inds][:, 2:4]
    #         bbox_target = self.bbox_coder.decode(priors[pos_inds, :2], coded_box)
    #     foreground_mask = torch.zeros_like(cls_target).to(torch.bool)
    #     foreground_mask[pos_inds] = 1
    #     return (foreground_mask,
    #             pos_ious, cls_target, bbox_target, num_pos_per_img)

    # def _get_targets_single(
    #         self, gt_instances: InstanceData, points: Tensor,
    #         regress_ranges: Tensor,
    #         num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
    #     """Compute regression and classification targets for a single image."""
    #     num_points = points.size(0)
    #     num_gts = len(gt_instances)
    #     gt_bboxes = gt_instances.bboxes
    #     gt_labels = gt_instances.labels
    #
    #     if num_gts == 0:
    #         return gt_labels.new_full((num_points,), self.num_classes), \
    #                gt_bboxes.new_zeros((num_points, 4)), \
    #                gt_bboxes.new_zeros((num_points, 1))
    #
    #     areas = gt_bboxes.areas
    #     gt_bboxes = gt_bboxes.regularize_boxes(self.angle_version)
    #
    #     # TODO: figure out why these two are different
    #     # areas = areas[None].expand(num_points, num_gts)
    #     areas = areas[None].repeat(num_points, 1)
    #     regress_ranges = regress_ranges[:, None, :].expand(
    #         num_points, num_gts, 2)
    #     points = points[:, None, :].expand(num_points, num_gts, 2)
    #     gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
    #     gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)
    #
    #     cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
    #     rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
    #                            dim=-1).reshape(num_points, num_gts, 2, 2)
    #     offset = points - gt_ctr
    #     offset = torch.matmul(rot_matrix, offset[..., None])
    #     offset = offset.squeeze(-1)
    #
    #     w, h = gt_wh[..., 0], gt_wh[..., 1]
    #     offset_x, offset_y = offset[..., 0], offset[..., 1]
    #     left = w / 2 + offset_x
    #     right = w / 2 - offset_x
    #     top = h / 2 + offset_y
    #     bottom = h / 2 - offset_y
    #     bbox_targets = torch.stack((left, top, right, bottom), -1)
    #
    #     # condition1: inside a gt bbox
    #     inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
    #     if self.center_sampling:
    #         # condition1: inside a `center bbox`
    #         radius = self.center_sample_radius
    #         stride = offset.new_zeros(offset.shape)
    #
    #         # project the points on current lvl back to the `original` sizes
    #         lvl_begin = 0
    #         for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
    #             lvl_end = lvl_begin + num_points_lvl
    #             stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
    #             lvl_begin = lvl_end
    #
    #         inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
    #         inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
    #                                                 inside_gt_bbox_mask)
    #
    #     # condition2: limit the regression range for each location
    #     max_regress_distance = bbox_targets.max(-1)[0]
    #     inside_regress_range = (
    #             (max_regress_distance >= regress_ranges[..., 0])
    #             & (max_regress_distance <= regress_ranges[..., 1]))
    #
    #     # if there are still more than one objects for a location,
    #     # we choose the one with minimal area
    #     areas[inside_gt_bbox_mask == 0] = INF
    #     areas[inside_regress_range == 0] = INF
    #     min_area, min_area_inds = areas.min(dim=1)
    #
    #     labels = gt_labels[min_area_inds]
    #     labels[min_area == INF] = self.num_classes  # set as BG
    #     bbox_targets = bbox_targets[range(num_points), min_area_inds]
    #     angle_targets = gt_angle[range(num_points), min_area_inds]
    #
    #     return labels, bbox_targets, angle_targets
