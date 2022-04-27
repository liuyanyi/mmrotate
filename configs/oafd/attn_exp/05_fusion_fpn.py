_base_ = 'base.py'

model = dict(
    # backbone=dict(
    #     plugins=[
    #         dict(
    #             cfg=dict(type='CSA'),
    #             stages=(False, True, True, True),
    #             position='after_conv3',
    #         ),
    #         dict(
    #             cfg=dict(type='EDSA'),
    #             stages=(False, True, True, True),
    #             position='after_conv3',
    #         )
    #     ]
    # )
    neck=[
        dict(
            type='FPN',
            in_channels=[32, 64, 160, 384],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5,
            relu_before_extra_convs=True),
        dict(
            type='ScaleFusion',
            in_channels=256,
        )
    ])

work_dir = './work_dirs/attn_exp/05'
