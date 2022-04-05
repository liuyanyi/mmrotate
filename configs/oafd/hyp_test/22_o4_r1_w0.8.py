_base_ = 'base.py'

model = dict(
    bbox_head=dict(
        angel_coder=dict(omega=4, radius=1),
        loss_angle=dict(loss_weight=0.8),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)))

work_dir = './work_dirs/hyp_test/22'
