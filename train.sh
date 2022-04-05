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

# python tools/train.py configs/oafd/hyp_test/06_o8_r6_w1.6.py
# python tools/train.py configs/oafd/hyp_test/07_o8_r5_w1.6.py
# python tools/train.py configs/oafd/hyp_test/08_o8_r4_w1.6.py
python tools/train.py configs/oafd/hyp_test/09_o8_r3_w1.6.py
python tools/train.py configs/oafd/hyp_test/10_o8_r2_w1.6.py
python tools/train.py configs/oafd/hyp_test/11_o2_r6_w0.4.py
python tools/train.py configs/oafd/hyp_test/12_o2_r5_w0.4.py
python tools/train.py configs/oafd/hyp_test/13_o2_r4_w0.4.py
python tools/train.py configs/oafd/hyp_test/14_o2_r3_w0.4.py
python tools/train.py configs/oafd/hyp_test/15_o2_r2_w0.4.py
