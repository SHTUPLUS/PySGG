#!/bin/bash
export OMP_NUM_THREADS=1
export gpu_num=4
export CUDA_VISIBLE_DEVICES="0,1,2,3"

exp_name="BGNN-3-3-learnable_scaling"


python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
       tools/relation_train_net.py \
       --config-file "configs/e2e_relBGNN_vg.yaml" \
       DEBUG False\
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
       MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR 0.13 \
       MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE 1.6 \
       EXPERIMENT_NAME "$exp_name" \
        SOLVER.IMS_PER_BATCH $[3*$gpu_num] \
        TEST.IMS_PER_BATCH $[$gpu_num] \
        SOLVER.VAL_PERIOD 2000 \
       MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/detection/pretrained_faster_rcnn/model_final.pth \
        SOLVER.CHECKPOINT_PERIOD 2000 


