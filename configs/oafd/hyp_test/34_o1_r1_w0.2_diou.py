_base_ = 'base.py'

model = dict(
    bbox_head=dict(
        angel_coder=dict(omega=1, radius=1),
        loss_angle=dict(loss_weight=0.2),
        loss_bbox=dict(type='DIoULoss', loss_weight=1.0)))

work_dir = './work_dirs/hyp_test/34'
