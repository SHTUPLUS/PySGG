# Model Zoo

## Evaluation Metrics
We follow the same evaluation protocols and metrics with [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/METRICS.md)(Visual Genome dataset) and [Graphical Contrastive Losses for Scene Graph Parsing](https://github.com/NVIDIA/ContrastiveLosses4VRD) (Openimage V4/V6)


## Reported Results

Here we list the SGGen results can be produced by our codebase with X-101-FPN backbone.
We reimplement those methods according the official implementation released by author.

∗ denotes the LVIS resampling is applied for this model.

### VG
| Model(SGGen) | mR@50 | mR@100 | R@50 | R@100 | head | body | tail |
|--------------|:-----:|:------:|:----:|:-----:|------|------|------|
| RelDN        |  6.0  |  7.3   | 31.4 |  35.9 | 34.1 | 6.6 | 1.1  |
| Motifs       |  5.5 |  6.8 | 32.1 |  36.9 | 36.1 | 7.0| 0.0  |
| Motifs∗      |  7.7 |  9.4 | 31.7 |  35.8 | 34.2 | 8.6 | 2.1 |
| VCTree       |  10.9 |  13.5 | 29.8 |  34.6 | -| - | -| 
| G-RCNN       |  5.8 |  6.7 | 29.78 |  32.8 | 28.6 | 6.5 | 0.1  |
| MSDN         |  6.1 |  7.2 | 31.9  |  36.6 | 35.1 | 5.5 | 0.0  |
| Unbiased     |  9.3 |  11.1 | 19.4 |  23.2 | 24.5 | 13.9 | 0.1  |
| GPS-Net      |  6.79 |  8.6 | 31.1 |  35.9 | 34.5 | 7.0 | 1.0  |
| GPS-Net*     |  7.4 |  9.5 | 27.8 |  32.1 | 30.4 | 8.5 | 3.8  |
| BGNN         |  10.9 |  13.55 | 29.8 |  34.6 | 33.4 | 13.4 | 6.4  |


### OIv6
| Model(SGGen) | mR@50 | R@50 | wmAP_rel | wmAP_phr | score_wtd |
|---|:---:|:---:|:---:|:---:|---|
| RelDN | 33.98 | 73.08 | 32.16 | 33.39 | 40.84
| RelDN* | 37.20 | 75.34 | 33.21 | 34.31 | 41.97
| VCTree | 33.91 | 74.08 | 34.16 | 33.11 | 40.21
| G-RCNN | 34.04 | 74.51 | 33.15 | 34.21 | 41.84
| Motifs | 32.68 | 71.63 | 29.91 | 31.59 | 38.93
| Unbiased | 35.47 | 69.30 | 30.74 | 32.80 | 39.27
| GPS-Net | 35.26 | 74.81 | 32.85 |  33.98 | 41.69
| GPS-Net* | 38.93 |  74.74 | 32.77 | 33.87 | 41.60
| BGNN | 41.71 | 74.96 | 33.83 | 34.87 | 42.47 |

