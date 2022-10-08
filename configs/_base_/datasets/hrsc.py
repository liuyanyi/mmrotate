# dataset settings
dataset_type = 'HRSCDataset'
data_root = '/home/wangchen/liuyanyi/datasets/hrsc/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        classwise=False,
        ann_file=data_root + 'ImageSets/trainval.txt',
        ann_subdir=data_root + 'FullDataSet/Annotations/',
        img_subdir=data_root + 'FullDataSet/AllImages/',
        # img_prefix=data_root + 'FullDataSet/AllImages/',
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        classwise=False,
        ann_file=data_root + 'ImageSets/test.txt',
        ann_subdir=data_root + 'FullDataSet/Annotations/',
        img_subdir=data_root + 'FullDataSet/AllImages/',
        # img_prefix=data_root + 'FullDataSet/AllImages/',
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='Box2SegmCocoMetric',
    metric=['bbox', 'segm'],
    # metric='mAP',
    # iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
)
# val_evaluator = dict(
#     type='DOTAMetric',
#     metric='mAP',
#     eval_mode='areas',
#     iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
test_evaluator = val_evaluator
