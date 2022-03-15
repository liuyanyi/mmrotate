import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from mmrotate.models.dense_heads.rotated_csl_retina_head import RotatedCSLRetinaHead


def main():
    head = RotatedCSLRetinaHead(num_classes=15,
                                in_channels=256,
                                label_type='csl',
                                label_mode='triangle',
                                omega=1,
                                radius=6,
                                angle_version='le90')

    angel_target = torch.tensor([[math.pi / 2]])

    encode = head.circular_encode(angel_target)

    decode = head.circular_decode(encode)

    encode_np = encode[0].numpy()
    x = np.arange(-90, 90)
    plt.plot(x, encode_np)
    plt.show()
    print('decode-target=', decode-angel_target)

if __name__ == '__main__':
    main()
