# A Toolkit for Scene Graph Benchmark in Pytorch(PySGG)

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.4.0-%237732a8)

Our paper [Bipartite Graph Network with Adaptive Message Passing for Unbiased Scene Graph Generation](https://arxiv.org/abs/2104.00308) has been accepted by CVPR 2021.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Model Zoo 
BGNN performance:
### VG
| Model(SGGen) | mR@50 | mR@100 | R@50 | R@100 | head | body | tail |
|--------------|:-----:|:------:|:----:|:-----:|------|------|------|
| BGNN         |  10.9 |  13.55 | 29.8 |  34.6 | 33.4 | 13.4 | 6.4  |


### OIv6
| Model(SGGen) | mR@50 | R@50 | wmAP_rel | wmAP_phr | score_wtd |
|---|:---:|:---:|:---:|:---:|---|
| BGNN | 41.71 | 74.96 | 33.83 | 34.87 | 42.47 |

The methods implemented in our toolkit and reported results are given in [Model Zoo.md](MODELZOO.md)

## Training **(IMPORTANT)**

### Prepare Faster-RCNN Detector
- You can download the pretrained Faster R-CNN we used in the paper: 
  - [VG](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EQIy64T-EK9Er9y8kVCDaukB79gJwfSsEIbey9g0Xag6lg?e=wkKHJs), 
  - [OIv6](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EfGXxc9byEtEnYFwd0xdlYEBcUuFXBjYxNUXVGkgc-jkfQ?e=lSlqnz), 
  - [OIv4](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EVWy0xJRx8RNo-zHF5bdANMBTYt6NvAaA59U32o426bRqw?e=iPVc0O) 
- put the checkpoint into the folder:
```
mkdir -p checkpoints/detection/pretrained_faster_rcnn/
# for VG
mv /path/vg_faster_det.pth checkpoints/detection/pretrained_faster_rcnn/
```

Then, you need to modify the pretrained weight parameter `MODEL.PRETRAINED_DETECTOR_CKPT` in configs yaml `configs/e2e_relBGNN_vg-oiv6-oiv4.yaml` to the path of corresponding pretrained rcnn weight to make sure you load the detection weight parameter correctly.



### Scene Graph Generation Model
You can follow the following instructions to train your own, which takes 4 GPUs for train each SGG model. The results should be very close to the reported results given in paper.

We provide the one-click script for training our BGNN model( in `scripts/rel_train_BGNN_[vg/oiv6/oiv4].sh`)
or you can copy the following command to train
```
gpu_num=4 && python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
       tools/relation_train_net.py \
       --config-file "configs/e2e_relBGNN_vg.yaml" \
        DEBUG False \
        EXPERIMENT_NAME "BGNN-3-3" \
        SOLVER.IMS_PER_BATCH $[3*$gpu_num] \
        TEST.IMS_PER_BATCH $[$gpu_num] \
        SOLVER.VAL_PERIOD 3000 \
        SOLVER.CHECKPOINT_PERIOD 3000 

```
We also provide the trained model pth of [BGNN(vg)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/Ee4PdxluTphEicUDckJIfmEBisAyUgkjeuerN_rjrG1CIw?e=pgr8a5),[BGNN(oiv6)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EdKOrWAOf4hMiDWbR3CgYrMB9w7ZwWul-Wc6IUSbs51Idw?e=oEEHIQ)



## Test
Similarly, we also provide the `rel_test.sh` for directly produce the results from the checkpoint provide by us.
By replacing the parameter of `MODEL.WEIGHT` to the trained model weight and selected dataset name in `DATASETS.TEST`, you can directly eval the model on validation or test set.


## Citations

If you find this project helps your research, please kindly consider citing our papers in your publications.

```
@InProceedings{Li_2021_CVPR,
    author    = {Li, Rongjie and Zhang, Songyang and Wan, Bo and He, Xuming},
    title     = {Bipartite Graph Network With Adaptive Message Passing for Unbiased Scene Graph Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {11109-11119}
}
```


## Acknowledgment
This repository is developed on top of the scene graph benchmarking framwork develped by [KaihuaTang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
