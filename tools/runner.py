import argparse
import logging
import os
import random
import time

import gpustat
import torch
import numpy as np


def start():
    mem = None
    gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])

    while True:
        info = gpustat.core.GPUStatCollection.new_query()
        gpu_info = info.jsonify()['gpus'][gpu_id]
        u_ratio = gpu_info['utilization.gpu']
        mem_ratio = gpu_info['memory.used'] / gpu_info['memory.total']
        # print("add meme")
        if mem is None:
            mem = torch.rand((25000, 8196), device=torch.torch.device("cuda"))

        if u_ratio < 30:
            if  mem_ratio < 0.50 :
                    mem = torch.cat((mem, torch.rand((25000, 8196), device=torch.torch.device("cuda")))).cuda()
            elif mem_ratio < 0.95 :
                mem = torch.cat((mem, torch.rand((10000, 8196), device=torch.torch.device("cuda")))).cuda()

            else:
                if mem is not None:
                    for _ in range(100):
                        mem *= mem
                        mem /= mem
                    time.sleep(0.001)



if __name__ == "__main__":
    start()