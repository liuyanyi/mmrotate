# Copyright (c) OpenMMLab. All rights reserved.
import json
from argparse import ArgumentParser

from mmcv import Config
from mmdet.apis import inference_detector, init_detector
from mmdet.datasets import build_dataset

import mmrotate  # noqa: F401
from mmrotate import obb2poly_np

colors = ['#FF8000', '#937374', '#0000FF', '#00FF00', '#A52A2A',
          '#FF00FF', '#FFFACD', '#FFC1C1', '#FF0000', '#8A2BE2',
          '#BDB76B', '#00FFFF', '#008B8B', '#003399', '#FFFF00', ]


def main():
    """Test a single image."""
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('output', help='Output image')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    # build the model from a config file and a checkpoint file
    cfg.data.test.test_mode = True

    model = init_detector(args.config, args.checkpoint, device=args.device)
    dataset = build_dataset(cfg.data.test)
    classes = dataset.CLASSES
    # test a single image
    result = inference_detector(model, args.img)
    result_dict = {}
    for name, color, res in zip(classes, colors, result):
        res_poly = obb2poly_np(res, dataset.version).tolist()
        result_dict[name] = {
            'num': len(res_poly),
            'color': color,
            'bbox': res_poly,
        }

    result_json = json.dumps(result_dict)

    with open("./demo/test.json", 'w+', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False)
    # show the results
    model.show_result(
        args.img, result, score_thr=args.score_thr, out_file=args.output)


if __name__ == '__main__':
    main()
