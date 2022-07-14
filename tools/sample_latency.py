# -*- coding: utf-8 -*-
import sys
import torch
import numpy as np
import scipy.io as io

sys.path.append("..")
from models import *
from utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import warnings
warnings.filterwarnings("ignore")

def measure_latency(model, inputs, N=20):
    """latency measurements"""

    model = model.eval()

    latency_list = []
    with torch.no_grad():
        # warm up
        for i in range(5):
            _ = model(inputs)
        for i in range(N):
            start = time.time()
            _ = model(inputs)
            if inputs.is_cuda:
                torch.cuda.synchronize()
            latency_list.append(time.time() - start)
    latency_list.sort()

    return sum(latency_list[:int(N*2/3)]) / int(N*2/3) * 1000 # ms

results = {}
arch_configs = io.loadmat("../data/arch_configs.mat")["arch_configs"]

for i, arch_config in enumerate(arch_configs[:20]):
    arch_config = arch_config.tolist()
    arch_config = [op.strip() for op in arch_config]
    inputs = torch.randn(8, 3, 224, 224).cuda()
    model = Network(arch_config=arch_config, channel_layout="mobilenetv2")
    model = model.cuda().eval()
    latency = measure_latency(model, inputs, N=6)
    print(i, latency)

arch_configs_sampled = []
latency_list = []

for i, arch_config in enumerate(arch_configs):
    arch_config = arch_config.tolist()
    arch_config = [op.strip() for op in arch_config]
    arch_configs_sampled.append(arch_config)
    inputs = torch.randn(8, 3, 224, 224).cuda()
    model = Network(arch_config=arch_config, channel_layout="mobilenetv2")
    model = model.cuda().eval()
    latency = measure_latency(model, inputs, N=6)
    print(i, latency)
    latency_list.append(latency)

results["latency_list"] = latency_list
results["arch_configs"] = arch_configs_sampled
io.savemat("../data/latency-sampling-results-bs-8.mat", results)
