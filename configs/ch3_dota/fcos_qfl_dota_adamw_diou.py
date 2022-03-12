_base_ = './base_fcos_dota.py'

# model settings
model = dict(
    bbox_head=dict(
        type='RotatedFCOSGFLHead',
        loss_bbox=dict(type='PolyDIoULoss', loss_weight=1.0),
    )
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001 / 4, weight_decay=0.0001)

custom_hooks = [dict(type='GFLModeSwitchHook', start_epochs=3)]
