# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core import BaseBBoxCoder

from mmrotate.core.bbox.transforms import norm_angle
from ..builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class DistanceAnglePointBBoxCoder(BaseBBoxCoder):
    """Distance Angle Point BBox coder.

    This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left,
    right) and decode it back to the original.

    Args:
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(self, clip_border=True, angle_range='oc'):
        super(BaseBBoxCoder, self).__init__()
        self.clip_border = clip_border
        self.angle_range = angle_range

    def encode(self, points, gt_bboxes, max_dis=None, eps=0.1):
        """Encode bounding box to distances.

        Args:
            points (Tensor): Shape (N, 2), The format is [x, y].
            gt_bboxes (Tensor): Shape (N, 4), The format is "xyxy"
            max_dis (float): Upper bound of the distance. Default None.
            eps (float): a small value to ensure target < max_dis, instead <=.
                Default 0.1.

        Returns:
            Tensor: Box transformation deltas. The shape is (N, 4).
        """
        assert points.size(0) == gt_bboxes.size(0)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 5
        return self.obb2distance(points, gt_bboxes, max_dis, eps,
                                 self.angle_range)

    def decode(self, points, pred_bboxes, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2).
            pred_bboxes (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom). Shape (B, N, 4)
                or (N, 4)
            max_shape (Sequence[int] or torch.Tensor or Sequence[
                Sequence[int]],optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W). If priors shape is (B, N, 4), then
                the max_shape should be a Sequence[Sequence[int]],
                and the length of max_shape should also be B.
                Default None.
        Returns:
            Tensor: Boxes with shape (N, 4) or (B, N, 4)
        """
        assert points.size(0) == pred_bboxes.size(0)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 5
        if self.clip_border is False:
            max_shape = None
        return self.distance2obb(points, pred_bboxes, max_shape,
                                 self.angle_range)

    def obb2distance(self,
                     points,
                     distance,
                     max_dis=None,
                     eps=None,
                     angle_range='oc'):
        ctr, wh, thetas = torch.split(distance, [2, 2, 1], dim=1)

        Cos, Sin = torch.cos(thetas), torch.sin(thetas)
        Matrix = torch.cat([Cos, Sin, -Sin, Cos], dim=-1).reshape(-1, 2, 2)
        offset = points - ctr
        offset = torch.matmul(Matrix, offset[..., None])
        offset = offset.squeeze(-1)

        W, H = wh[..., 0], wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = W / 2 + offset_x
        right = W / 2 - offset_x
        top = H / 2 + offset_y
        bottom = H / 2 - offset_y
        return torch.stack((left, top, right, bottom, thetas.squeeze(-1)), -1)

    def distance2obb(self, points, distance, max_shape=None, angle_range='oc'):
        distance, theta = distance.split([4, 1], dim=1)

        Cos, Sin = torch.cos(theta), torch.sin(theta)
        Matrix = torch.cat([Cos, -Sin, Sin, Cos], dim=1).reshape(-1, 2, 2)

        wh = distance[:, :2] + distance[:, 2:]
        offset_t = (distance[:, 2:] - distance[:, :2]) / 2
        offset_t = offset_t.unsqueeze(2)
        offset = torch.bmm(Matrix, offset_t).squeeze(2)
        ctr = points + offset

        obbs = torch.cat([ctr, wh, theta], dim=1)

        x, y, w, h, theta = obbs.unbind(dim=-1)
        w_regular = torch.where(w > h, w, h)
        h_regular = torch.where(w > h, h, w)
        theta_regular = torch.where(w > h, theta, theta + torch.pi / 2)
        theta_regular = norm_angle(theta_regular, angle_range)
        w_regular = torch.clamp(w_regular, min=0.001)
        h_regular = torch.clamp(h_regular, min=0.001)
        return torch.stack([x, y, w_regular, h_regular, theta_regular], dim=-1)
