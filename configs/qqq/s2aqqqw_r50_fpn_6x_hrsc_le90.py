_base_ = [
    '../_base_/datasets/hrsc.py', '../_base_/schedules/schedule_6x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le90'
model = dict(
    type='WRSS',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head_init=dict(
        type='QQQHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
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
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            frm_cfg=dict(
                type='WeightedAlignConv',
                feat_channels=256,
                kernel_size=3,
                strides=[8, 16, 32, 64, 128]),
            # anchor_generator=dict(
            #     type='PseudoRotatedAnchorGenerator',
            #     strides=[8, 16, 32, 64, 128]),
            anchor_generator=dict(
                type='PseudoRotatedAnchorGenerator',
                # angle_version=angle_version,
                # scales=[4],
                # ratios=[1.0],
                strides=[8, 16, 32, 64, 128]),
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

file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomRotate', prob=0.5, angle_range=180),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

custom_hooks = [dict(type='QQQHook', start_epoch=6)]
# optim_wrapper = dict(type='AmpOptimWrapper')

optim_wrapper = dict(
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
