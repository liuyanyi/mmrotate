data = dict(test=dict(type='DOTADataset', pipeline=[...], version='le135'))

model = dict(
    type='OAFD',
    backbone=dict(type='ResNet'),
    neck=dict(type='FPN'),
    bbox_head=dict(type='OAFDHead'),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))
