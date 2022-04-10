_base_ = ['./rotated_retinanet_obb_r50_fpn_1x_dota_le90.py']

fp16 = dict(loss_scale='dynamic')

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(_delete_=True, type='PolyIoULoss', loss_weight=1.0)))
