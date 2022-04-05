_base_ = 'base.py'

model = dict(
    neck=dict(
        _delete_=True,
        type='YOLOXPAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_csp_blocks=1),
    bbox_head=dict(
        strides=[8, 16, 32, 64],
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 1E8)),
        angel_coder=dict(omega=1, radius=1),
        loss_angle=dict(loss_weight=0.2),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)))

work_dir = './work_dirs/hyp_test/b3'
