#!/bin/bash

export OMP_NUM_THREADS=1
export gpu_num=4
export CUDA_VISIBLE_DEVICES="0,1,2,3"


archive_dir="/group/rongjie/projects/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet-BGNNPredictor/(2021-06-19_13)BGNN-3-3-no-resample-minmax"

python -m torch.distributed.launch --master_port 10029 --nproc_per_node=$gpu_num  \
  tools/relation_test_net.py \
  --config-file "configs/e2e_relBGNN_vg.yaml"\
    TEST.IMS_PER_BATCH $[$gpu_num] \
   MODEL.WEIGHT  "model.pth"\
   MODEL.ROI_RELATION_HEAD.EVALUATE_REL_PROPOSAL False \
   DATASETS.TEST "('VG_stanford_filtered_with_attribute_test', )"

