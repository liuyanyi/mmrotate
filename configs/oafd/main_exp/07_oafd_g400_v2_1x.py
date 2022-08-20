_base_ = 'base.py'

model = dict(
    backbone=dict(
        _delete_=True,
        type='RegNet',
        arch='regnetx_400mf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://regnetx_400mf')),
    neck=dict(
        type='FPN',
        in_channels=[32, 64, 160, 384],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=3,
        relu_before_extra_convs=True),
    bbox_head=dict(
        strides=[8, 16, 32],
        regress_ranges=((-1, 64), (64, 128), (128, 1e8)),
    ),
)
work_dir = './work_dirs/main_exp/small_v2_debug'
