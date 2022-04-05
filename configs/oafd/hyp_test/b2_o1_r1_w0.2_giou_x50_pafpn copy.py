_base_ = 'base.py'

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
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext50_32x4d')),
    neck=dict(type='PAFPN', ),
    bbox_head=dict(
        angel_coder=dict(omega=1, radius=1),
        loss_angle=dict(loss_weight=0.2),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)))

work_dir = './work_dirs/hyp_test/b2'
