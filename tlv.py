# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from collections import Sequence
from pathlib import Path

import mmcv
import numpy as np
from mmcv import Config, DictAction
from mmdet.datasets.builder import build_dataset

from mmrotate.datasets import DOTADataset
from mmrotate.core.visualization import imshow_det_rbboxes


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
    instance_num = 0
    img_num = len(dataset)
    obj_ratios = []
    sizes = []
    small_num = 0
    large_num = 0
    cls_num = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 0,
        13: 0,
        14: 0,
    }
    # i=0
    for index, item in enumerate(dataset):
        gt_labels = item['gt_labels']
        for gt_label in gt_labels:
            cls_num[gt_label] += 1
        gt_bboxes = item['gt_bboxes']
        for i in range(len(gt_labels)):
            box = gt_bboxes[i]
            # size = box[2] * box[3]
            # if size <= 90000:
            #     small_num += 1
            # else:
            #     large_num += 1
            # sizes.append(size)
            r1 = box[2] / box[3]
            r2 = box[3] / box[2]
            obj_ratios.append(np.maximum(r1, r2))
        # instance_num += gt_labels.size
        # i += 1
        # if i>100:
        #     break
        # if index>500:
        #     break
        progress_bar.update()

    # 直方图
    import matplotlib.pyplot as plt
    plt.rc('font', family='Times New Roman')

    # 调整figsize
    # plt.figure(figsize=(8, 4))
    # ax = plt.subplot(111)
    # sizes = np.array(sizes)
    # b = list(range(0, 20000, 200))
    # ax.hist(sizes, bins=b)
    # # ax.set_yscale('log')
    # plt.show()

    plt.figure(figsize=(8, 4))
    ax = plt.subplot(111)
    b2 = np.arange(1, 15, 1)
    obj_ratios = np.array(obj_ratios)
    ax.hist(obj_ratios, bins=b2, density=True)
    # ax.set_yscale('log')

    plt.show()

    # plt.figure(figsize=(8, 4))
    # ax = plt.subplot(111)
    # plt.subplots_adjust(left=0.25)
    # ax.barh(list(DOTADataset.CLASSES), list(cls_num.values()))
    # plt.xticks(rotation=30)
    # plt.show()
    print('\n')
    print(cls_num)
    print('small_num:', small_num)
    print('large_num:', large_num)
    print(0)


if __name__ == '__main__':
    main()
