_base_ = 'main_r50.py'

model = dict(
    backbone=dict(plugins=[
        dict(
            cfg=dict(type='ECA'),
            stages=(False, True, True, True),
            position='after_conv3',
        )
    ]))

work_dir = './work_dirs/attn/01_r50_eca'
