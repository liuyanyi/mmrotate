# dataset settings
dataset_type = 'HRSCDataset'
data_root = '/root/autodl-tmp/dataset/hrsc/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='HRResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_heads'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classwise=False,
        with_head=True,
        ann_file=data_root + 'ImageSets/trainval.txt',
        ann_subdir=data_root + 'FullDataSet/Annotations/',
        img_subdir=data_root + 'FullDataSet/AllImages/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classwise=False,
        with_head=True,
        ann_file=data_root + 'ImageSets/test.txt',
        ann_subdir=data_root + 'FullDataSet/Annotations/',
        img_subdir=data_root + 'FullDataSet/AllImages/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classwise=False,
        with_head=True,
        ann_file=data_root + 'ImageSets/test.txt',
        ann_subdir=data_root + 'FullDataSet/Annotations/',
        img_subdir=data_root + 'FullDataSet/AllImages/',
        pipeline=test_pipeline))
