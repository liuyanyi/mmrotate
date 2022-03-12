#python tools/train.py configs/ch3_dota/base_ret_dota.py --work-dir=./work_dirs/ch3_dota/01_baseline
#python tools/test.py configs/ch3_dota/base_ret_dota.py ./work_dirs/ch3_dota/01_baseline/latest.pth --format-only --eval-options submission_dir=./work_dirs/ch3_dota/01_baseline/result nproc=1
#
#python tools/train.py configs/ch3_dota/ret_dota_adamw.py --work-dir=./work_dirs/ch3_dota/02_ret_adamw
#python tools/test.py configs/ch3_dota/ret_dota_adamw.py ./work_dirs/ch3_dota/02_ret_adamw/latest.pth --format-only --eval-options submission_dir=./work_dirs/ch3_dota/02_ret_adamw/result nproc=1
#
#python tools/train.py configs/ch3_dota/fcos_dota_adamw.py --work-dir=./work_dirs/ch3_dota/03_fcos
#python tools/test.py configs/ch3_dota/fcos_dota_adamw.py ./work_dirs/ch3_dota/03_fcos/latest.pth --format-only --eval-options submission_dir=./work_dirs/ch3_dota/03_fcos/result nproc=1

python tools/train.py configs/ch3_dota/fcos_dota_adamw_diou.py --work-dir=./work_dirs/ch3_dota/04_fcos_diou
python tools/test.py configs/ch3_dota/fcos_dota_adamw_diou.py ./work_dirs/ch3_dota/04_fcos_diou/latest.pth --format-only --eval-options submission_dir=./work_dirs/ch3_dota/04_fcos_diou/result nproc=1

python tools/train.py configs/ch3_dota/fcos_qfl_dota_adamw_diou.py --work-dir=./work_dirs/ch3_dota/05_fcos_qfl_diou
python tools/test.py configs/ch3_dota/fcos_qfl_dota_adamw_diou.py ./work_dirs/ch3_dota/05_fcos_qfl_diou/latest.pth --format-only --eval-options submission_dir=./work_dirs/ch3_dota/05_fcos_qfl_diou/result nproc=1
