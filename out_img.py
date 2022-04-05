import os
from argparse import ArgumentParser

from mmdet.apis import init_detector, show_result_pyplot

from mmrotate.apis import inference_detector_by_patches


def main():
    # out_path = './work_dirs/test_show/'
    out_path = './work_dirs/out_imgs/'
    # build the model from a config file and a checkpoint file
    models = {}
    imgs = [
        '/home/wangchen/liuyanyi/datasets/dota/test/images/P0006__1024__0___0.png',
        '/home/wangchen/liuyanyi/datasets/dota/test/images/P0992__1024__824___824.png',
        '/home/wangchen/liuyanyi/datasets/dota/test/images/P2588__1024__267___0.png',
        '/home/wangchen/liuyanyi/datasets/dota/test/images/P0031__1024__3296___2771.png'
    ]
    names = ['oafd_baseline', 'retinanet', 'roi_trans']
    cfgs = [
        './work_dirs/ms_exp/01/01_oafd_r50_1x.py',
        './work_dirs/rotated_retinanet_hbb_r50_fpn_1x_dota_le90/rotated_retinanet_hbb_r50_fpn_1x_dota_le90.py',
        './configs/roi_trans/roi_trans_r50_fpn_1x_dota_le90.py'
    ]
    ckpts = [
        './work_dirs/ms_exp/01/latest.pth',
        './work_dirs/rotated_retinanet_hbb_r50_fpn_1x_dota_le90/latest.pth',
        './work_dirs/ckpt/roi_trans_r50_fpn_1x_dota_le90-d1f0b77a.pth'
    ]

    for name, cfg, ckpt in zip(names, cfgs, ckpts):
        models[name] = init_detector(cfg, ckpt, device='cuda:0')

    for img in imgs:
        for name in names:
            model = models[name]
            result = inference_detector_by_patches(model, img, [1024], [524],
                                                   [1.0], 0.1)
            filename = os.path.basename(img)
            f_name, suffix = os.path.splitext(filename)
            out_name = f_name + '_' + name + '.' + suffix
            path = os.path.join(out_path, out_name)
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
                mask_color=None,
                out_file=path)


if __name__ == '__main__':
    main()
