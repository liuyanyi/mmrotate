# Copyright (c) OpenMMLab. All rights reserved.
import math

import cv2
import mmcv
import numpy as np
from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val

from .single_stage import RotatedSingleStageDetector
from ..builder import ROTATED_DETECTORS


@ROTATED_DETECTORS.register_module()
class RHRetinaNet(RotatedSingleStageDetector):
    """Implementation of Rotated `RetinaNet.

    <https://arxiv.org/abs/1708.02002>`_
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RHRetinaNet,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained, init_cfg)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_heads,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_heads,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        bbox_results = [
            self.rbbox2result(det_bboxes, det_heads, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels, det_heads in bbox_list
        ]
        return bbox_results

    def rbbox2result(self, bboxes, heads, labels, num_classes):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): shape (n, 6)
            labels (torch.Tensor): shape (n, )
            num_classes (int): class number, including background class

        Returns:
            list(ndarray): bbox results of each class
        """
        if bboxes.shape[0] == 0:
            return [(np.zeros((0, 6), dtype=np.float32), np.zeros((0), dtype=np.int64)) for _ in range(num_classes)]
        else:
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
            heads = heads.cpu().numpy()
            return [(bboxes[labels == i, :], heads[labels == i]) for i in range(num_classes)]

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(226, 43, 138),
                    text_color='white',
                    thickness=2,
                    font_scale=0.25,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    **kwargs):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (torch.Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()

        bbox_result, head_result = tuple(zip(*result))

        bboxes = np.vstack(bbox_result)
        heads = np.vstack(head_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        imshow_det_rbboxes(
            img,
            bboxes,
            heads,
            labels,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img


def imshow_det_rbboxes(img,
                       bboxes,
                       heads,
                       labels,
                       class_names=None,
                       score_thr=0.3,
                       bbox_color=(226, 43, 138),
                       text_color='white',
                       thickness=2,
                       font_scale=0.25,
                       show=True,
                       win_name='',
                       wait_time=0,
                       out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 5) or
            (n, 6).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes is not None and bboxes.ndim == 2
    assert labels.ndim == 1

    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 6
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = (226, 43,
                  138) if bbox_color is None else color_val(bbox_color)
    head_color = (0, 255, 0)
    text_color = (255, 255,
                  255) if text_color is None else color_val(text_color)
    for bbox, head, label in zip(bboxes, heads, labels):
        xc, yc, w, h, ag = bbox[:5]
        score = bbox[5] if bboxes.shape[1] == 6 else None
        wx, wy = w / 2 * math.cos(ag), w / 2 * math.sin(ag)
        hx, hy = -h / 2 * math.sin(ag), h / 2 * math.cos(ag)
        p1 = (xc - wx - hx, yc - wy - hy)
        p2 = (xc + wx - hx, yc + wy - hy)
        p3 = (xc + wx + hx, yc + wy + hy)
        p4 = (xc - wx + hx, yc - wy + hy)
        ps = np.int0(np.array([p1, p2, p3, p4]))
        cv2.drawContours(img, [ps], -1, bbox_color, thickness=thickness)

        if head[0] == 0:
            head = np.int0(0.5 * (np.array(p4) + np.array(p3)))
        elif head[0] == 1:
            head = np.int0(0.5 * (np.array(p2) + np.array(p3)))
        elif head[0] == 2:
            head = np.int0(0.5 * (np.array(p1) + np.array(p2)))
        elif head[0] == 3:
            head = np.int0(0.5 * (np.array(p1) + np.array(p4)))
        cv2.circle(img, head, 4, head_color, 10)
        # cv2.circle(img, np.int0((921, 596)), 4, (0, 0, 255), 10)

        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if score:
            label_text += '|{:.02f}'.format(score)
        font = cv2.FONT_HERSHEY_COMPLEX
        text_size = cv2.getTextSize(label_text, font, font_scale, 1)
        text_width = text_size[0][0]
        text_height = text_size[0][1]
        cv2.rectangle(img, (int(xc), int(yc) - text_height - 2),
                      (int(xc) + text_width, int(yc) + 3), (0, 128, 0), -1)
        cv2.putText(img, label_text, (int(xc), int(yc)), font, font_scale,
                    text_color, 1)
        # cv2.putText(img, 'p1', np.int0(p1), font, font_scale,
        #             text_color, 1)
        # cv2.putText(img, 'p2', np.int0(p2), font, font_scale,
        #             text_color, 1)
        # cv2.putText(img, 'p3', np.int0(p3), font, font_scale,
        #             text_color, 1)
        # cv2.putText(img, 'p4', np.int0(p4), font, font_scale,
        #             text_color, 1)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)

    return img


def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    if wait_time == 0:  # prevent from hanging if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)
