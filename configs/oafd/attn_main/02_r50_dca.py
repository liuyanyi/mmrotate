_base_ = 'main_r50.py'

model = dict(
    backbone=dict(plugins=[
        dict(
            cfg=dict(type='DCA'),
            stages=(False, True, True, True),
            position='after_conv3',
        )
    ]))

work_dir = './work_dirs/attn/02_r50_dca'
