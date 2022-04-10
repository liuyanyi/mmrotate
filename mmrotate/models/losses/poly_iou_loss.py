# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import torch
import torch.nn as nn
from mmcv.ops import diff_iou_rotated_2d
from mmcv.ops.diff_iou_rotated import (box2corners_th, calculate_area,
                                       sort_indices)
from mmdet.models.losses.utils import weighted_loss

from ..builder import ROTATED_LOSSES


def poly_enclose(pts1, pts2):
    all_pts = torch.cat([pts1, pts2], dim=1)
    mask1 = pts1.new_ones((pts1.size(0), pts1.size(1)))
    mask2 = pts2.new_ones((pts2.size(0), pts2.size(1)))
    masks = torch.cat([mask1, mask2], dim=1)
    return all_pts, masks


def convex_areas(pred, target):
    pred_pts = box2corners_th(pred)
    target_pts = box2corners_th(target)

    enclose_pts, enclose_masks = poly_enclose(pred_pts, target_pts)
    index = sort_indices(enclose_pts, enclose_masks)
    return calculate_area(index, enclose_pts)


def get_bbox_areas(bboxes):
    return bboxes[..., 2] * bboxes[..., 3]


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def poly_iou_loss(pred, target, linear=False, mode='log', eps=1e-6):
    """Poly IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x, y, h, w, angle),
            shape (n, 5).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn(
            'DeprecationWarning: Setting "linear=True" in '
            'poly_iou_loss is deprecated, please use "mode=`linear`" '
            'instead.')

    ious = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0))
    ious = ious.squeeze(0).clamp(min=eps)

    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious**2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def poly_giou_loss(pred, target, eps=1e-6):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.
    Args:
        pred (torch.Tensor): Predicted bboxes of format (x, y, h, w, angle),
            shape (n, 5).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 5).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    areas1, areas2 = get_bbox_areas(pred), get_bbox_areas(target)
    overlap = diff_iou_rotated_2d(pred, target)

    union = areas1 + areas2 - overlap + eps
    ious = (overlap / union).clamp(min=eps)

    enclose_areas = convex_areas(pred, target)

    gious = ious - (enclose_areas - union) / enclose_areas
    loss = 1 - gious
    return loss


@ROTATED_LOSSES.register_module()
class PolyIoULoss(nn.Module):
    """PolyIoULoss.

    Computing the Poly IoU loss between a set of predicted rbboxes
    and target rbboxes.
    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0,
                 mode='log'):
        super(PolyIoULoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in '
                          'IOULoss is deprecated, please use "mode=`linear`" '
                          'instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 5) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * poly_iou_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@ROTATED_LOSSES.register_module()
class PolyGIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(PolyGIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * poly_giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
