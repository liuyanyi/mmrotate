_base_ = [
    '../_base_/datasets/hrsc.py', '../_base_/schedules/schedule_6x.py',
    '../_base_/default_runtime.py'
]
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'  # noqa

angle_version = 'le90'
model = dict(
    type='WRSS',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='mmdet.SiLU')),
    neck=dict(
        type='mmdet.CSPNeXtPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='mmdet.SiLU')),
    bbox_head_init=dict(
        type='QQQHead',
        num_classes=1,
        in_channels=128,
        stacked_convs=1,
        feat_channels=128,
        strides=[8, 16, 32],
        regress_ranges=((-1, 64), (64, 128), (128, 1e8)),
        # center_sampling=False,
        # center_sample_radius=1.5,
        # norm_on_bbox=True,
        # separate_angle=False,
        # scale_angle=True,
        # bbox_coder=dict(
        #     type='DistanceAnglePointCoder', angle_version=angle_version),
        center_sampling=False,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        separate_angle=False,
        scale_angle=False,
        angle_version=angle_version,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        # angle_coder=dict(
        #     type='CSLCoder',
        #     angle_version=angle_version,
        #     omega=1,
        #     window='gaussian',
        #     radius=1.5),
        # angle_coder=dict(
        #     type='DCLCoder',
        #     angle_version=angle_version,
        #     encode_size=7),
        reg_max=16,
        angle_coder=dict(type='DistributionAngleCoder', reg_max=16),
        # loss_angle=dict(
        #     type='SmoothFocalLoss', gamma=2.0, alpha=0.25, loss_weight=10.0),
        loss_angle=dict(
            type='mmdet.DistributionFocalLoss',
            # reduction='sum',
            loss_weight=0.20),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            # reduction='sum',
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(
            type='RotatedIoULoss',
            # reduction='sum',
            loss_weight=1.0)),
    bbox_head_refine=[
        dict(
            type='MAWS2ASepBNRefineHead',
            num_classes=1,
            in_channels=128,
            stacked_convs=1,
            feat_channels=128,
            frm_cfg=dict(
                type='WeightedAlignConv',
                feat_channels=128,
                kernel_size=3,
                strides=[8, 16, 32]),
            # anchor_generator=dict(
            #     type='PseudoRotatedAnchorGenerator',
            #     strides=[8, 16, 32, 64, 128]),
            anchor_generator=dict(
                type='PseudoRotatedAnchorGenerator',
                # angle_version=angle_version,
                # scales=[4],
                # ratios=[1.0],
                strides=[8, 16, 32]),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            # loss_cls=dict(
            #     type='mmdet.QualityFocalLoss',
            #     use_sigmoid=True,
            #     beta=2.0,
            #     loss_weight=1.0),
            # reg_decoded_bbox=True,
            # loss_bbox=dict(
            #     type='RotatedIoULoss', loss_weight=1.0),
            reg_decoded_bbox=False,
            loss_bbox=dict(type='mmdet.L1Loss', loss_weight=1.0),
            # reg_max=10,
            # angle_coder=dict(type='DistributionScaleAngleCoder', reg_max=10),
            # loss_angle=dict(
            #     type='mmdet.DistributionFocalLoss', loss_weight=0.20),
        )
    ],
    train_cfg=dict(
        init=None,
        refine=[
            dict(
                # assigner=dict(
                #     type='RotatedATSSAssigner',
                #     iou_calculator=dict(type='RBboxOverlaps2D'),
                #     topk=9),
                assigner=dict(
                    type='mmdet.MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        stage_loss_weights=[1.0]),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.CachedMosaic', img_scale=(800, 800), pad_val=114.0),
    dict(
        type='RandomResize',
        resize_type='mmdet.Resize',
        scale=(1600, 1600),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=(800, 800)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.Pad', size=(800, 800), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.CachedMixUp',
        img_scale=(800, 800),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='RandomResize',
        resize_type='mmdet.Resize',
        scale=(800, 800),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=(800, 800)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.Pad', size=(800, 800), pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

# test_pipeline = [
#     dict(
#         type='LoadImageFromFile',
#         file_client_args={{_base_.file_client_args}}),
#     dict(type='Resize', scale=(640, 640), keep_ratio=True),
#     dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    batch_sampler=None,
    pin_memory=True,
    dataset=dict(pipeline=train_pipeline))
# val_dataloader = dict(
#     batch_size=5, num_workers=10, dataset=dict(pipeline=test_pipeline))
# test_dataloader = val_dataloader

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

max_epochs = 300
stage2_num_epochs = 20
base_lr = 0.004 / 4
interval = 10

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 10)])

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]
