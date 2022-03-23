# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
from collections import Sequence
from pathlib import Path

import cv2
import mmcv
import numpy as np
from mmcv import Config, DictAction
from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val
from mmrotate.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type, cfg_options):
    """Retrieve the dataset config file.

    Args:
        config_path (str): Path of the config file.
        skip_type (list[str]): List of the useless pipeline to skip.
        cfg_options (dict): dict of configs to merge from.
    """

    def skip_pipeline_steps(config):
        config['pipeline'] = [
            x for x in config.pipeline if x['type'] not in skip_type
        ]

    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg and train_data_cfg[
        'type'] != 'MultiImageMixDataset':
        train_data_cfg = train_data_cfg['dataset']

    if isinstance(train_data_cfg, Sequence):
        [skip_pipeline_steps(c) for c in train_data_cfg]
    else:
        skip_pipeline_steps(train_data_cfg)

    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)

    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))

    for item in dataset:
        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None

        gt_bboxes = item['gt_bboxes']
        gt_labels = item['gt_labels']
        gt_heads = item['gt_heads']
        gt_heads_ori = item['gt_heads_ori']
        # print(gt_heads_ori)

        imshow_det_rbboxes(
            item['img'],
            gt_bboxes,
            gt_heads,
            gt_labels,
            class_names=dataset.CLASSES,
            score_thr=0,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=filename,
            bbox_color=dataset.PALETTE,
            text_color=(200, 200, 200))

        progress_bar.update()


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

        mid_points = []
        mid_points.append(np.array((xc - hx, yc - hy)))
        mid_points.append(np.array((xc + hx, yc + hy)))
        mid_points.append(np.array((xc - wx, yc - wy)))
        mid_points.append(np.array((xc + wx, yc + wy)))

        cv2.circle(img, np.int0(mid_points[head]), 4, head_color, 10)

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
        cv2.putText(img, 'p1', np.int0(p1), font, font_scale,
                    text_color, 1)
        cv2.putText(img, 'p2', np.int0(p2), font, font_scale,
                    text_color, 1)
        cv2.putText(img, 'p3', np.int0(p3), font, font_scale,
                    text_color, 1)
        cv2.putText(img, 'p4', np.int0(p4), font, font_scale,
                    text_color, 1)
        cv2.putText(img, str(head), np.int0(mid_points[head]), font, font_scale,
                    text_color, 1)

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


if __name__ == '__main__':
    main()
