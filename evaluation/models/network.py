# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import math
import random
import numpy as np

from .operations import *

def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Network(nn.Module):
    """stand-alone network"""
    def __init__(self, arch_config=None, width_mult=1.0, dropout=0.2, num_classes=1000, affine=True, track_running_stats=True):
        super(Network, self).__init__()

        # channel layout infos
        first_conv = 16
        last_conv = 1984
        block_infos = [
            # in_channels, out_channels, H_in, W_in, stride
            [first_conv, 16, 112, 112, 1], # 1-st stage
            [16, 24, 112, 112, 2], # 2-nd stage
            [24, 24, 56, 56, 1],
            [24, 48, 56, 56, 2], # 3-rd stage
            [48, 48, 28, 28, 1],
            [48, 48, 28, 28, 1],
            [48, 72, 28, 28, 2], # 4-th stage
            [72, 72, 14, 14, 1],
            [72, 72, 14, 14, 1],
            [72, 72, 14, 14, 1],
            [72, 128, 14, 14, 1], # 5-th stage
            [128, 128, 14, 14, 1],
            [128, 128, 14, 14, 1],
            [128, 160, 14, 14, 2], # 6-th stage
            [160, 160, 7, 7, 1],
            [160, 160, 7, 7, 1],
            [160, 160, 7, 7, 1],
            [160, 176, 7, 7, 1], # 7-th stage
            [176, 176, 7, 7, 1],
            [176, 176, 7, 7, 1],
            [176, 176, 7, 7, 1],
            [176, 384, 7, 7, 1], # 8-th stage
        ]

        block_infos_after_mult = []
        for info in block_infos:
            in_channels, out_channels, H_in, W_in, stride = info
            in_channels = make_divisible(in_channels * width_mult)
            out_channels = make_divisible(out_channels * width_mult)
            block_infos_after_mult.append([in_channels, out_channels, H_in, W_in, stride])

        # first conv
        first_conv = make_divisible(first_conv * width_mult)
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, first_conv, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(first_conv, affine=affine, track_running_stats=track_running_stats),
            Activation("relu6")
        )

        # intermediate blocks
        assert arch_config[0] == "MBK3E1"
        assert "MBK3E1" not in arch_config[1:]
        assert len(arch_config) == 22
        blocks = []
        for i, info in enumerate(block_infos_after_mult):
            in_channels, out_channels, H_in, W_in, stride = info
            blocks.append(ops_all[arch_config[i]](in_channels, out_channels, stride, affine, track_running_stats))
        self.blocks = nn.Sequential(*blocks)

        # last conv
        last_conv = make_divisible(last_conv * width_mult)
        self.last_conv = nn.Sequential(
            nn.Conv2d(out_channels, last_conv, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_conv, affine=affine, track_running_stats=track_running_stats),
            Activation("relu6"),
            nn.AdaptiveAvgPool2d(1)
        )

        # final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(last_conv, 1280, bias=False),
            nn.BatchNorm1d(1280),
            Activation("relu6"),
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes, bias=True)
        )

        self.init_weights()
        self.set_bn_param(0.1, 0.001)

    def init_weights(self, model_init='he_fout', init_div_groups=True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def set_bn_param(self, bn_momentum, bn_eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum
                m.eps = bn_eps
        return

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.last_conv(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)

        return x

# if __name__ == "__main__":
#     for i in range(10):
#         op_names = [name for name in ops_all]
#         arch_config = [op_names[0]] + [op_names[random.randint(1, len(op_names)-1)] for _ in range(21)]
#         model = Network(arch_config=arch_config, width_mult=1.0)
#         inputs = torch.randn(1, 3, 224, 224)
#         print(model(inputs).size())
