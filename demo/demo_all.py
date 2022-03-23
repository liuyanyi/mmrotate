# Copyright (c) OpenMMLab. All rights reserved.
"""Inference on single image.

Example:
```
python demo/image_demo.py \
    demo/demo.jpg \
    configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_v3.py \
    work_dirs/oriented_rcnn_r50_fpn_1x_dota_v3/epoch_12.pth \
    demo/vis.jpg
```
"""  # nowq
import os
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector

import mmrotate  # noqa: F401


def main():
    """Test a single image."""
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    path = "/root/autodl-tmp/dataset/hrsc/FullDataSet/AllImages"
    out_path = './work_dirs/alldet2c/'
    for file in os.listdir(path):
        file_path = os.path.join(path, file)

        # test a single image
        result = inference_detector(model, file_path)
        # show the results
        model.show_result(
            file_path, result, score_thr=args.score_thr, out_file=out_path+file)


if __name__ == '__main__':
    main()
