import os
from argparse import ArgumentParser

import mmcv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from tools.misc.browse_dataset import retrieve_data_cfg

from mmrotate import build_dataset
from mmrotate.apis import inference_detector_by_patches


def main():
    data_cfg = '/home/wangchen/liuyanyi/mmrotate/work_dirs/hrsc_exp/oafd_g400/oafd_g400.py'
    cfg = retrieve_data_cfg(data_cfg,
                            ['DefaultFormatBundle', 'Normalize', 'Collect'],
                            None)

    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))

    # out_path = './work_dirs/test_show/'
    out_path = './work_dirs/hrsc_out/models/'
    # build the model from a config file and a checkpoint file
    models = {}
    # imgs = [
    #     '/home/wangchen/liuyanyi/datasets/dota/test/images/P0006__1024__0___0.png',
    #     '/home/wangchen/liuyanyi/datasets/dota/test/images/P0992__1024__824___824.png',
    #     '/home/wangchen/liuyanyi/datasets/dota/test/images/P2588__1024__267___0.png',
    #     '/home/wangchen/liuyanyi/datasets/dota/test/images/P0031__1024__3296___2771.png'
    # ]
    names = ['oafds']
    cfgs = [
        '/home/wangchen/liuyanyi/mmrotate/work_dirs/hrsc_exp/oafd_g400/oafd_g400.py',
        # '/home/wangchen/liuyanyi/mmrotate/work_dirs/vhr/mask/mrcnn.py',
    ]
    ckpts = [
        '/home/wangchen/liuyanyi/mmrotate/work_dirs/hrsc_exp/oafd_g400/latest.pth',
        # '/home/wangchen/liuyanyi/mmrotate/work_dirs/vhr/mask/latest.pth',
    ]

    for name, cfg, ckpt in zip(names, cfgs, ckpts):
        models[name] = init_detector(cfg, ckpt, device='cuda:0')

    for item in dataset:
        img = item.get('filename')
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
                bbox_color='hrsc',
                text_color='hrsc',
                mask_color='hrsc',
                out_file=path)

        progress_bar.update()


if __name__ == '__main__':
    main()
