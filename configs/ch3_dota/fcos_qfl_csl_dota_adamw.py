_base_ = './base_fcos_dota.py'

# model settings
model = dict(
    bbox_head=dict(
        type='RotatedFCOSGFLCSLHead',
        angel_coder=dict(
            type='CSLCoder',
            angle_version='le90',
            omega=1,
            window='gaussian',
            radius=2.5),
        loss_angle=dict(
            type='SmoothFocalLoss', gamma=2.0, alpha=0.25, loss_weight=0.5),
    ))

optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001 / 4, weight_decay=0.0001)

custom_hooks = [dict(type='GFLModeSwitchHook', start_epochs=3)]
