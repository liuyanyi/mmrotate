_base_ = 'base.py'

pretrain = 'https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth'

model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=50,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=pretrain)), )
work_dir = './work_dirs/main_exp/04'
