#python tools/train.py configs/ch3_dota/base_ret_dota.py --work-dir=./work_dirs/ch3_dota/01_baseline
#python tools/test.py configs/ch3_dota/base_ret_dota.py ./work_dirs/ch3_dota/01_baseline/latest.pth --format-only --eval-options submission_dir=./work_dirs/ch3_dota/01_baseline/result nproc=1
#
#python tools/train.py configs/ch3_dota/ret_dota_adamw.py --work-dir=./work_dirs/ch3_dota/02_ret_adamw
#python tools/test.py configs/ch3_dota/ret_dota_adamw.py ./work_dirs/ch3_dota/02_ret_adamw/latest.pth --format-only --eval-options submission_dir=./work_dirs/ch3_dota/02_ret_adamw/result nproc=1
#
#python tools/train.py configs/ch3_dota/fcos_dota_adamw.py --work-dir=./work_dirs/ch3_dota/03_fcos
#python tools/test.py configs/ch3_dota/fcos_dota_adamw.py ./work_dirs/ch3_dota/03_fcos/latest.pth --format-only --eval-options submission_dir=./work_dirs/ch3_dota/03_fcos/result nproc=1

#python tools/train.py configs/ch3_dota/fcos_dota_adamw_diou.py --work-dir=./work_dirs/ch3_dota/04_fcos_diou
#python tools/test.py configs/ch3_dota/fcos_dota_adamw_diou.py ./work_dirs/ch3_dota/04_fcos_diou/latest.pth --format-only --eval-options submission_dir=./work_dirs/ch3_dota/04_fcos_diou/result nproc=1
#
#python tools/train.py configs/ch3_dota/fcos_qfl_dota_adamw_diou.py --work-dir=./work_dirs/ch3_dota/05_fcos_qfl_diou
#python tools/test.py configs/ch3_dota/fcos_qfl_dota_adamw_diou.py ./work_dirs/ch3_dota/05_fcos_qfl_diou/latest.pth --format-only --eval-options submission_dir=./work_dirs/ch3_dota/05_fcos_qfl_diou/result nproc=1

# python tools/train.py configs/oafd/hyp_test/16_o1_r6_w0.2.py
# python tools/train.py configs/oafd/hyp_test/17_o1_r5_w0.2.py
# python tools/train.py configs/oafd/hyp_test/18_o1_r4_w0.2.py
# python tools/train.py configs/oafd/hyp_test/19_o1_r3_w0.2.py
# python tools/train.py configs/oafd/hyp_test/20_o1_r2_w0.2.py


# python tools/train.py configs/oafd/hyp_test/21_o8_r1_w1.6.py
# python tools/train.py configs/oafd/hyp_test/22_o4_r1_w0.8.py
# python tools/train.py configs/oafd/hyp_test/23_o2_r1_w0.4.py
# python tools/train.py configs/oafd/hyp_test/24_o1_r1_w0.2.py
# python tools/train.py configs/oafd/hyp_test/25_o8_r0_w1.6.py
# python tools/train.py configs/oafd/hyp_test/26_o4_r0_w0.8.py
# python tools/train.py configs/oafd/hyp_test/27_o2_r0_w0.4.py
# python tools/train.py configs/oafd/hyp_test/28_o1_r0_w0.4.py

# python tools/train.py configs/oafd/hyp_test/25_o8_r0_w1.6.py
# python tools/train.py configs/oafd/hyp_test/29_o1_r1_w0.1.py
# python tools/train.py configs/oafd/hyp_test/30_o1_r1_w0.3.py
# python tools/train.py configs/oafd/hyp_test/31_o1_r1_w0.4.py

# python tools/train.py configs/oafd/hyp_test/32_o1_r1_w0.5.py
# python tools/train.py configs/oafd/hyp_test/33_o1_r1_w0.2_iou.py
# python tools/train.py configs/oafd/hyp_test/34_o1_r1_w0.2_diou.py
# python tools/train.py configs/oafd/hyp_test/35_o1_r1_w0.2_ciou.py

# python tools/train.py configs/oafd/hyp_test/b1_o1_r1_w0.2_giou_pafpn.py
# python tools/train.py configs/oafd/hyp_test/b2_o1_r1_w0.2_giou_x50_pafpn.py

python tools/train.py configs/oafd/main_exp/01_oafd_r50_1x.py
python tools/train.py configs/oafd/main_exp/02_oafd_effb0_1x.py
python tools/train.py configs/oafd/main_exp/03_oafd_swint_1x.py
python tools/train.py configs/oafd/main_exp/04_oafd_x50_1x.py
