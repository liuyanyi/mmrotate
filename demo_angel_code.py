import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from mmrotate.models.utils.angel_coder import build_angle_coder


def main():
    # head = RotatedCSLRetinaHead(num_classes=15,
    #                             in_channels=256,
    #                             label_type='csl',
    #                             label_mode='gaussian',
    #                             omega=1,
    #                             radius=6,
    #                             angle_version='le90')
    cfg = dict(
        type='CSLCoder',
        angle_version='le135',
        omega=1,
        window='gaussian',
        radius=6)

    coder = build_angle_coder(cfg)

    angel_target = torch.tensor([[math.pi / 4]])

    encode = coder.encode(angel_target)

    decode = coder.decode(encode)

    encode_np = encode[0].numpy()
    x = np.arange(0, 180) - coder.angle_offset
    plt.plot(x, encode_np)
    plt.show()
    print('decode-target=', decode - angel_target)


if __name__ == '__main__':
    main()
