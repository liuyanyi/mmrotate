_base_ = 'main_400.py'

model = dict(
    backbone=dict(plugins=[
        dict(
            cfg=dict(type='DSA'),
            stages=(False, True, True, True),
            position='after_conv3',
        )
    ]))

work_dir = './work_dirs/attn/06_g400_dsa'
