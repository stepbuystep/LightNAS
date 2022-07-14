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

from read_power import *
def measure_power(model, inputs, N=20, device="xavier"):
    """power measurements"""

    model = model.eval()

    power_list = []
    with torch.no_grad():
        # warm up
        for i in range(5):
            _ = model(inputs)
        for i in range(N):
            _ = model(inputs)
            if inputs.is_cuda:
                torch.cuda.synchronize()
            power = eval("read_power_%s"%device)()
            power_list.append(power)
    power_list.sort()
    results = power_list[:int(2 * N / 3)]
    return sum(results) / len(results)

results = {}
arch_configs = io.loadmat("../data/arch_configs.mat")["arch_configs"]

for i, arch_config in enumerate(arch_configs[:50]):
    arch_config = arch_config.tolist()
    arch_config = [op.strip() for op in arch_config]
    inputs = torch.randn(8, 3, 224, 224).cuda()
    model = Network(arch_config=arch_config, channel_layout="mobilenetv2")
    model = model.cuda().eval()
    power = measure_power(model, inputs, N=6)
    print(i, power)

arch_configs_sampled = []
power_list = []

for i, arch_config in enumerate(arch_configs):
    arch_config = arch_config.tolist()
    arch_config = [op.strip() for op in arch_config]
    arch_configs_sampled.append(arch_config)
    inputs = torch.randn(8, 3, 224, 224).cuda()
    model = Network(arch_config=arch_config, channel_layout="mobilenetv2")
    model = model.cuda().eval()
    power = measure_power(model, inputs, N=6)
    print(i, power)
    power_list.append(power)

results["power_list"] = power_list
results["arch_configs"] = arch_configs_sampled
io.savemat("../data/power-sampling-results-bs-8.mat", results)
