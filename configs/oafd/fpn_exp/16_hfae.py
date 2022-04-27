_base_ = 'main_r50.py'

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(neck=[
    dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    dict(
        type='ScaleFusion',
        in_channels=256,
        out_channels=256,
        num_blocks=1,
        name='HFA_E')
])

work_dir = './work_dirs/fpn_exp/16_HFAE'
