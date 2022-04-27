# python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/oafd/hyp_test/29_o1_r1_w0.1.py work_dirs/hyp_test/29/latest.pth --launcher pytorch
# python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py /home/wangchen/liuyanyi/mmrotate/work_dirs/s2anet_r50_fpn_1x_dota_le135/s2anet_r50_fpn_1x_dota_le135.py /home/wangchen/liuyanyi/mmrotate/work_dirs/s2anet_r50_fpn_1x_dota_le135/latest.pth --launcher pytorch
#python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py /home/wangchen/liuyanyi/mmrotate/work_dirs/rotated_retinanet_obb_r50_fpn_1x_dota_le90/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py /home/wangchen/liuyanyi/mmrotate/work_dirs/rotated_retinanet_obb_r50_fpn_1x_dota_le90/latest.pth --launcher pytorch


#python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/hrsc/oafd_baseline.py work_dirs/hrsc_exp/oafd_baseline/latest.pth --launcher pytorch
#python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/hrsc/oafd_g400.py work_dirs/hrsc_exp/oafd_g400/latest.pth --launcher pytorch
#python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/hrsc/oriented_rcnn_r50_fpn_3x_hrsc_le90.py work_dirs/hrsc_exp/oriented_rcnn_r50_fpn_3x_hrsc_le90/latest.pth --launcher pytorch
#python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/hrsc/r3det_r50_fpn_3x_hrsc_oc.py work_dirs/hrsc_exp/r3det_r50_fpn_3x_hrsc_oc/latest.pth --launcher pytorch
#python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/hrsc/roi_trans_r50_fpn_3x_hrsc_le90.py work_dirs/hrsc_exp/roi_trans_r50_fpn_3x_hrsc_le90/latest.pth --launcher pytorch
#python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/hrsc/rotated_faster_rcnn_r50_fpn_3x_hrsc_le90.py work_dirs/hrsc_exp/rotated_faster_rcnn_r50_fpn_3x_hrsc_le90/latest.pth --launcher pytorch
#python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/hrsc/rotated_retinanet_hbb_r50_fpn_3x_hrsc_le90.py work_dirs/hrsc_exp/rotated_retinanet_hbb_r50_fpn_3x_hrsc_le90/latest.pth --launcher pytorch
#python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/hrsc/rotated_retinanet_obb_r50_fpn_3x_hrsc_le90.py work_dirs/hrsc_exp/rotated_retinanet_obb_r50_fpn_3x_hrsc_le90/latest.pth --launcher pytorch
#python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/hrsc/s2anet_r50_fpn_3x_hrsc_le135.py work_dirs/hrsc_exp/s2anet_r50_fpn_3x_hrsc_le135/latest.pth --launcher pytorch

#python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/sffpn/roi_trans_r50_sffpn_1x_dota_le90.py work_dirs/sff_exp/dota/roi_trans/latest.pth --launcher pytorch
#python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/sffpn/rotated_retinanet_obb_r50_sffpn_1x_dota_le90.py work_dirs/sff_exp/dota/rreto/latest.pth --launcher pytorch
#python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/sffpn/oafdb_dota_sffpn.py work_dirs/sff_exp/dota/oafdb/latest.pth --launcher pytorch
#python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/sffpn/oafds_dota_sffpn.py work_dirs/sff_exp/dota/oafds/latest.pth --launcher pytorch
#python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/sffpn/oafdb_dota_sffpn.py work_dirs/sff_exp/dota/oafdb_ms/latest.pth --launcher pytorch
#python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/sffpn/oafds_dota_sffpn.py work_dirs/sff_exp/dota/oafds_ms/latest.pth --launcher pytorch

python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/sffpn/roi_trans_r50_sffpn_3x_hrsc_le90.py work_dirs/sff_exp/hrsc/roi_trans/latest.pth --launcher pytorch
python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/sffpn/rotated_retinanet_obb_r50_sffpn_3x_hrsc_le90.py work_dirs/sff_exp/hrsc/rreto/latest.pth --launcher pytorch
python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/sffpn/rotated_retinanet_obb_r50_sffpn_6x_hrsc_le90.py work_dirs/sff_exp/hrsc/rreto_6x/latest.pth --launcher pytorch
python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/sffpn/rotated_retinanet_obb_r50_fpn_6x_hrsc_le90.py work_dirs/sff_exp/hrsc/rreto_6x_fpn/latest.pth --launcher pytorch
python -m torch.distributed.run --nproc_per_node=1 tools/analysis_tools/benchmark.py configs/sffpn/oafds_sffpn.py work_dirs/sff_exp/hrsc/oafds/latest.pth --launcher pytorch
