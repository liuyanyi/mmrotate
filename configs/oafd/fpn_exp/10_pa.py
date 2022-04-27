_base_ = 'main_r50.py'

model = dict(neck=dict(type='PAFPN'))

work_dir = './work_dirs/fpn_exp/10_PA'
