_base_ = ['./rotated_retinanet_obb_r50_fpn_3x_hrsc_le90.py']

angle_version = 'le90'
model = dict(bbox_head=dict(assign_by_circumhbbox=angle_version))
