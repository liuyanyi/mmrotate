_base_ = 'oafdb_sffpn.py'

model = dict(
    backbone=dict(
        _delete_=True,
        type='RegNet',
        arch='regnetx_400mf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://regnetx_400mf')),
    neck=[
        dict(
            type='FPN',
            in_channels=[32, 64, 160, 384],
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
            name='HFAB')
    ])
work_dir = './work_dirs/sff_exp/hrsc/oafds'
