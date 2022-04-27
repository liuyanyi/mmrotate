
# python tools/train.py configs/oafd/attn_exp/00_base.py
# python tools/train.py configs/oafd/attn_main/01_dca.py
# python tools/train.py configs/oafd/attn_main/02_eca.py
# python tools/train.py configs/oafd/attn_main/03_dsa.py
# python tools/train.py configs/oafd/attn_main/05_fusion_fpn.py

# python tools/train.py configs/oafd/attn_exp/03_edsa.py
# python tools/train.py configs/oafd/attn_exp/04_csa_edsa.py

# python tools/train.py configs/oafd/main_exp/02_oafd_effb0_1x.py
# python tools/train.py configs/oafd/main_exp/03_oafd_swint_1x.py
# python tools/train.py configs/oafd/main_exp/04_oafd_x50_1x.py
#python tools/train.py configs/oafd/attn_main/01_r50_eca.py
#python tools/train.py configs/oafd/attn_main/02_r50_dca.py
#python tools/train.py configs/oafd/attn_main/03_g400_eca.py
#python tools/train.py configs/oafd/attn_main/04_g400_dca.py
#python tools/train.py configs/oafd/attn_main/05_r50_dsa.py
#python tools/train.py configs/oafd/attn_main/06_g400_dsa.py
# python tools/train.py configs/oafd/fpn_exp/01_base.py
# python tools/train.py configs/oafd/fpn_exp/02_TA.py
# python tools/train.py configs/oafd/fpn_exp/03_TB.py
# python tools/train.py configs/oafd/fpn_exp/04_TC.py
# python tools/train.py configs/oafd/fpn_exp/05_TD.py
# python tools/train.py configs/oafd/fpn_exp/06_TE.py
# python tools/train.py configs/oafd/fpn_exp/07_TF.py

# python tools/train.py configs/oafd/fpn_exp/10_pa.py
#python tools/train.py configs/oafd/fpn_exp/01_base.py
#python tools/train.py configs/oafd/fpn_exp/12_dyhead.py

# python tools/train.py configs/hrsc/oafd_baseline.py --work-dir=./work_dirs/hrsc_exp/oafd_baseline
# python tools/train.py configs/hrsc/oafd_g400.py --work-dir=./work_dirs/hrsc_exp/oafd_g400
# python tools/train.py configs/hrsc/oriented_rcnn_r50_fpn_3x_hrsc_le90.py --work-dir=./work_dirs/hrsc_exp/oriented_rcnn_r50_fpn_3x_hrsc_le90
# python tools/train.py configs/hrsc/r3det_r50_fpn_3x_hrsc_oc.py --work-dir=./work_dirs/hrsc_exp/r3det_r50_fpn_3x_hrsc_oc
# python tools/train.py configs/hrsc/roi_trans_r50_fpn_3x_hrsc_le90.py --work-dir=./work_dirs/hrsc_exp/roi_trans_r50_fpn_3x_hrsc_le90
# python tools/train.py configs/hrsc/rotated_faster_rcnn_r50_fpn_3x_hrsc_le90.py --work-dir=./work_dirs/hrsc_exp/rotated_faster_rcnn_r50_fpn_3x_hrsc_le90
# python tools/train.py configs/hrsc/rotated_retinanet_hbb_r50_fpn_3x_hrsc_le90.py --work-dir=./work_dirs/hrsc_exp/rotated_retinanet_hbb_r50_fpn_3x_hrsc_le90
# python tools/train.py configs/hrsc/rotated_retinanet_obb_r50_fpn_3x_hrsc_le90.py --work-dir=./work_dirs/hrsc_exp/rotated_retinanet_obb_r50_fpn_3x_hrsc_le90
# python tools/train.py configs/hrsc/s2anet_r50_fpn_3x_hrsc_le135.py --work-dir=./work_dirs/hrsc_exp/s2anet_r50_fpn_3x_hrsc_le135


# python tools/train.py configs/oafd/fpn_exp/10_pa.py
# python tools/train.py configs/oafd/fpn_exp/11_nas.py
# python tools/train.py configs/oafd/fpn_exp/15_dyheadc.py
# python tools/train.py configs/oafd/fpn_exp/20_hfa.py
# python tools/train.py configs/oafd/fpn_exp/21_hfab.py
# python tools/train.py configs/oafd/fpn_exp/22_hfae.py
# python tools/train.py configs/oafd/fpn_exp/23_hfass.py
# python tools/train.py configs/oafd/fpn_exp/12_dyhead.py
# python tools/train.py configs/oafd/fpn_exp/14_dyheadb.py

# python tools/train.py configs/oafd/fpn_exp/20_hfa.py  --work-dir=./work_dirs/hfa_exp/20_3
# python tools/train.py configs/oafd/fpn_exp/21_hfab.py --work-dir=./work_dirs/hfa_exp/21_3
#python tools/train.py configs/oafd/fpn_exp/22_hfae.py --work-dir=./work_dirs/hfa_exp/22_3

# python tools/train.py configs/sffpn/oafdb_sffpn.py
# python tools/train.py configs/sffpn/oafds_sffpn.py
# python tools/train.py configs/sffpn/roi_trans_r50_sffpn_3x_hrsc_le90.py
#python tools/train.py configs/sffpn/s2anet_r50_sffpn_3x_hrsc_le135.py

# python tools/train.py configs/sffpn/oafdb_dota_sffpn.py
# python tools/train.py configs/sffpn/oafds_dota_sffpn.py
#python tools/train.py configs/sffpn/rotated_retinanet_obb_r50_sffpn_3x_hrsc_le90.py

python tools/train.py configs/sffpn/rotated_retinanet_obb_r50_sffpn_6x_hrsc_le90.py
python tools/train.py configs/sffpn/rotated_retinanet_obb_r50_fpn_6x_hrsc_le90.py

python tools/train.py configs/sffpn/oafdb_dota_sffpn.py
python tools/train.py configs/sffpn/oafds_dota_sffpn.py
python tools/train.py configs/sffpn/roi_trans_r50_sffpn_1x_dota_le90.py
python tools/train.py configs/sffpn/rotated_retinanet_obb_r50_sffpn_1x_dota_le90.py
