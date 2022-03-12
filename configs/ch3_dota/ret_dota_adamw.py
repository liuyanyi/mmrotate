_base_ = './base_ret_dota.py'

optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001 / 4, weight_decay=0.0001)
