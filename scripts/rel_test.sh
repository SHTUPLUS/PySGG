#!/bin/bash

export OMP_NUM_THREADS=1
export gpu_num=4
export CUDA_VISIBLE_DEVICES="0,1,2,3"


archive_dir="/group/rongjie/projects/Scene-Graph-Benchmark.pytorch/checkpoints/predcls-BGNNPredictor/(2021-07-16_08)BGNN-3-3-learnable_scaling(resampling)"

python -m torch.distributed.launch --master_port 10029 --nproc_per_node=$gpu_num  \
  tools/relation_test_net.py \
  --config-file "$archive_dir/config.yml"\
    TEST.IMS_PER_BATCH $[$gpu_num] \
   MODEL.WEIGHT  "$archive_dir/model_0020000.pth"\
   MODEL.ROI_RELATION_HEAD.EVALUATE_REL_PROPOSAL False \
   DATASETS.TEST "('VG_stanford_filtered_with_attribute_test', )"

