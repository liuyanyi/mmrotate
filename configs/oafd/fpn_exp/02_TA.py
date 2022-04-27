_base_ = 'main_r50.py'

model = dict(neck=[
    dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    dict(type='ScaleFusion', in_channels=256, out_channels=256, name='TA')
])

work_dir = './work_dirs/fpn_exp/02_TA'
