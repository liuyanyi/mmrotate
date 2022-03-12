_base_ = './s2anet_r50_fpn_1x_dota_le135.py'

fp16 = dict(loss_scale='dynamic')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001 / 4, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
