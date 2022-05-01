import os
from argparse import ArgumentParser

import mmcv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from tools.misc.browse_dataset import retrieve_data_cfg

from mmrotate import build_dataset
from mmrotate.apis import inference_detector_by_patches


def main():
    # data_cfg = '/home/wangchen/liuyanyi/mmrotate/work_dirs/hrsc_exp/oafd_g400/oafd_g400.py'
    # cfg = retrieve_data_cfg(data_cfg,
    #                         ['DefaultFormatBundle', 'Normalize', 'Collect'],
    #                         None)

    # dataset = build_dataset(cfg.data.train)

    # progress_bar = mmcv.ProgressBar(len(dataset))

    # out_path = './work_dirs/test_show/'
    out_path = './work_dirs/dota_out/models/'
    # build the model from a config file and a checkpoint file
    models = {}
    imgs = [
        # '/home/wangchen/liuyanyi/datasets/dota/test/images/P0016__1024__0___0.png',
        '/home/wangchen/liuyanyi/datasets/dota/test/images/P0031__1024__3296___2771.png',
        '/home/wangchen/liuyanyi/datasets/dota/test/images/P0006__1024__0___0.png',
        '/home/wangchen/liuyanyi/datasets/dota/test/images/P0992__1024__824___824.png',
        # '/home/wangchen/liuyanyi/datasets/dota/test/images/P0043__1024__1648___824.png',
        # '/home/wangchen/liuyanyi/datasets/dota/test/images/P0045__1024__0___824.png',
        '/home/wangchen/liuyanyi/datasets/dota/test/images/P0046__1024__0___0.png',
        # '/home/wangchen/liuyanyi/datasets/dota/test/images/P0051__1024__824___2472.png',
        # '/home/wangchen/liuyanyi/datasets/dota/test/images/P0059__1024__0___0.png',
        # '/home/wangchen/liuyanyi/datasets/dota/test/images/P0084__1024__0___824.png',
        # '/home/wangchen/liuyanyi/datasets/dota/test/images/P0137__1024__3296___1648.png',
        '/home/wangchen/liuyanyi/datasets/dota/test/images/P0138__1024__0___0.png',
        # '/home/wangchen/liuyanyi/datasets/dota/test/images/P0157__1024__100___190.png',
        # '/home/wangchen/liuyanyi/datasets/dota/test/images/P2707__1024__1648___0.png',
        '/home/wangchen/liuyanyi/datasets/dota/test/images/P2366__1024__0___680.png',
        # '/home/wangchen/liuyanyi/datasets/dota/test/images/P2024__1024__0___1648.png',
        # '/home/wangchen/liuyanyi/datasets/dota/test/images/P1796__1024__4120___2472.png',
        # '/home/wangchen/liuyanyi/datasets/harbor_01.png',
        # '/home/wangchen/liuyanyi/datasets/dota/test/images/',
    ]
    # names = ['roi_trans', 'retinanet', 'oafd']
    names = ['oafds']
    cfgs = [
        # '/home/wangchen/liuyanyi/mmrotate/configs/roi_trans/roi_trans_r50_fpn_1x_dota_le90.py',
        # '/home/wangchen/liuyanyi/mmrotate/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py',
        # '/home/wangchen/liuyanyi/mmrotate/work_dirs/ms_exp/01/01_oafd_r50_1x.py',
        '/home/wangchen/liuyanyi/mmrotate/work_dirs/ms_exp/05/05_oafd_g400_1x.py'
        # '/home/wangchen/liuyanyi/mmrotate/work_dirs/vhr/mask/mrcnn.py',
    ]
    ckpts = [
        # '/home/wangchen/liuyanyi/mmrotate/work_dirs/ckpt/roi_trans_r50_fpn_1x_dota_le90-d1f0b77a.pth',
        # '/home/wangchen/liuyanyi/mmrotate/work_dirs/ckpt/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth',
        # '/home/wangchen/liuyanyi/mmrotate/work_dirs/ms_exp/01/latest.pth',
        '/home/wangchen/liuyanyi/mmrotate/work_dirs/ms_exp/05/latest.pth',
        # '/home/wangchen/liuyanyi/mmrotate/work_dirs/vhr/mask/latest.pth',
    ]

    for name, cfg, ckpt in zip(names, cfgs, ckpts):
        models[name] = init_detector(cfg, ckpt, device='cuda:0')

    progress_bar = mmcv.ProgressBar(len(imgs))

    for img in imgs:
        # for item in dataset:
        # img = item.get('filename')
        for name in names:
            model = models[name]
            # result = inference_detector_by_patches(model, img, [1024], [524],
            #                                        [1.0], 0.1)
            result = inference_detector(model, img)
            filename = os.path.basename(img)
            path = os.path.join(out_path, name, filename)
            if hasattr(model, 'module'):
                model = model.module
            model.show_result(
                img,
                result,
                score_thr=0.3,
                show=True,
                wait_time=0,
                bbox_color='dota',
                text_color='dota',
                mask_color='dota',
                out_file=path)

        progress_bar.update()


if __name__ == '__main__':
    main()
