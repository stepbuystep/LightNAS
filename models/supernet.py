# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

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

def channel_shuffle(x, groups=4):
    """channel shuffle (see ShuffleNetV2)"""

    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels //  groups

    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x

class SuperBlock(nn.Module):
    """super block in supernet (single-path)"""
    def __init__(self, in_channels, out_channels, stride, affine=True, track_running_stats=True, op_names=None, groups=4):
        super(SuperBlock, self).__init__()

        if op_names is not None:
            self.op_names = op_names
        else:
            self.op_names = ["MBK3E3", "MBK3E6", "MBK5E3", "MBK5E6", "MBK7E3", "MBK7E6", "SkipConnect"]

        self.in_channels_tmp = in_channels // groups
        self.out_channels_tmp = out_channels - in_channels * (groups - 1) // groups
        self.groups = groups

        self.transform = None
        if in_channels != out_channels:
            self.transform = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=track_running_stats),
            )

        self.max_pool = None
        if stride != 1:
            self.max_pool = nn.MaxPool2d(2, 2)

        self.blocks = nn.ModuleList()
        for name in self.op_names:
            op = ops_all[name](self.in_channels_tmp, self.out_channels_tmp, stride, affine=affine, track_running_stats=track_running_stats)
            self.blocks.append(op)

    def forward(self, x, alphas, eta=0.):
        results = []
        alphas = alphas.view(-1, 1)
        assert alphas.numel() == len(self.op_names)

        if self.groups > 1:
            x1 = x[:, :self.in_channels_tmp, :, :]
            x2 = x[:, self.in_channels_tmp:, :, :]

            for alpha, block in zip(alphas, self.blocks):
                if alpha == 1:
                    results.append(alpha * block(x1))
                elif alpha == 0:
                    results.append(alpha)
                else:
                    raise ValueError("something wrong here")
            results = sum(results)

            if self.max_pool is None:
                results = torch.cat([results, x2], dim=1)
            else:
                results = torch.cat([results, self.max_pool(x2)], dim=1)
            results = channel_shuffle(results, self.groups)
        else:
            for alpha, block in zip(alphas, self.blocks):
                if alpha == 1:
                    results.append(alpha * block(x))
                elif alpha == 0:
                    results.append(alpha)
                else:
                    raise ValueError("something wrong here")
            results = sum(results)

        if eta == 0.:
            return results
        else:
            if self.transform is not None:
                x_aux = self.transform(x)
            else:
                x_aux = x
            if self.max_pool is not None:
                x_aux = self.max_pool(x_aux)
            return results + eta * x_aux

class SuperNetwork(nn.Module):
    """super network"""
    def __init__(self, width_mult=1.0, dropout=0.2, num_classes=1000, channel_layout="mobilenetv2", affine=True, track_running_stats=True, op_names=None, groups=4):
        super(SuperNetwork, self).__init__()

        if op_names is not None:
            self.op_names = op_names
        else:
            self.op_names = ["MBK3E3", "MBK3E6", "MBK5E3", "MBK5E6", "MBK7E3", "MBK7E6", "SkipConnect"]

        if channel_layout == "mobilenetv2":
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
        else:
            raise NotImplementedError

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

        # first block
        in_channels, out_channels, H_in, W_in, stride = block_infos_after_mult[0]
        self.first_block = ops_all["MBK3E1"](in_channels, out_channels, stride, affine=affine, track_running_stats=track_running_stats)
        # intermediate blocks
        self.blocks = nn.ModuleList()
        for info in block_infos_after_mult[1:]:
            in_channels, out_channels, H_in, W_in, stride = info
            self.blocks.append(SuperBlock(in_channels, out_channels, stride, affine, track_running_stats, self.op_names, groups))

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
            Activation("relu6"),
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes, bias=True)
        )

        self.init_weights()
        self.init_alphas()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_alphas(self, gpu=None):
        num_ops = len(self.op_names)
        num_layers = 21 # 22-1
        if gpu is None:
            self.alphas = Variable(1e-3 * torch.randn(num_layers, num_ops).float(), requires_grad=True)
        else:
            self.alphas = Variable(1e-3 * torch.randn(num_layers, num_ops).float().cuda(gpu), requires_grad=True)

    def parse_alphas(self):
        idxs = torch.argmax(self.alphas, dim=-1)
        arch_config = ["MBK3E1"]
        for idx in idxs:
            arch_config.append(self.op_names[idx])
        return arch_config

    def get_alphas(self):
        return self.alphas

    def forward(self, x, alphas_gumbels=None, eta=0.):
        x = self.first_conv(x)
        x = self.first_block(x)
        for i, block in enumerate(self.blocks):
            x = block(x, alphas_gumbels[i], eta)
        x = self.last_conv(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)

        return x
