# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np

ops_all = {
    "MBK3E1": lambda in_channels, out_channels, stride, affine, track_running_stats: InvertedResidual(in_channels, out_channels, 3, stride, expand_ratio=1, act_func="relu6", use_se=False, bias=False, affine=affine, track_running_stats=track_running_stats),
    "MBK3E3": lambda in_channels, out_channels, stride, affine, track_running_stats: InvertedResidual(in_channels, out_channels, 3, stride, expand_ratio=3, act_func="relu6", use_se=False, bias=False, affine=affine, track_running_stats=track_running_stats),
    "MBK3E6": lambda in_channels, out_channels, stride, affine, track_running_stats: InvertedResidual(in_channels, out_channels, 3, stride, expand_ratio=6, act_func="relu6", use_se=False, bias=False, affine=affine, track_running_stats=track_running_stats),
    "MBK5E3": lambda in_channels, out_channels, stride, affine, track_running_stats: InvertedResidual(in_channels, out_channels, 5, stride, expand_ratio=3, act_func="relu6", use_se=False, bias=False, affine=affine, track_running_stats=track_running_stats),
    "MBK5E6": lambda in_channels, out_channels, stride, affine, track_running_stats: InvertedResidual(in_channels, out_channels, 5, stride, expand_ratio=6, act_func="relu6", use_se=False, bias=False, affine=affine, track_running_stats=track_running_stats),
    "MBK7E3": lambda in_channels, out_channels, stride, affine, track_running_stats: InvertedResidual(in_channels, out_channels, 7, stride, expand_ratio=3, act_func="relu6", use_se=False, bias=False, affine=affine, track_running_stats=track_running_stats),
    "MBK7E6": lambda in_channels, out_channels, stride, affine, track_running_stats: InvertedResidual(in_channels, out_channels, 7, stride, expand_ratio=6, act_func="relu6", use_se=False, bias=False, affine=affine, track_running_stats=track_running_stats),
    "MBK3E1SE": lambda in_channels, out_channels, stride, affine, track_running_stats: InvertedResidual(in_channels, out_channels, 3, stride, expand_ratio=1, act_func="relu6", use_se=True, bias=False, affine=affine, track_running_stats=track_running_stats),
    "MBK3E3SE": lambda in_channels, out_channels, stride, affine, track_running_stats: InvertedResidual(in_channels, out_channels, 3, stride, expand_ratio=3, act_func="relu6", use_se=True, bias=False, affine=affine, track_running_stats=track_running_stats),
    "MBK3E6SE": lambda in_channels, out_channels, stride, affine, track_running_stats: InvertedResidual(in_channels, out_channels, 3, stride, expand_ratio=6, act_func="relu6", use_se=True, bias=False, affine=affine, track_running_stats=track_running_stats),
    "MBK5E3SE": lambda in_channels, out_channels, stride, affine, track_running_stats: InvertedResidual(in_channels, out_channels, 5, stride, expand_ratio=3, act_func="relu6", use_se=True, bias=False, affine=affine, track_running_stats=track_running_stats),
    "MBK5E6SE": lambda in_channels, out_channels, stride, affine, track_running_stats: InvertedResidual(in_channels, out_channels, 5, stride, expand_ratio=6, act_func="relu6", use_se=True, bias=False, affine=affine, track_running_stats=track_running_stats),
    "MBK7E3SE": lambda in_channels, out_channels, stride, affine, track_running_stats: InvertedResidual(in_channels, out_channels, 7, stride, expand_ratio=3, act_func="relu6", use_se=True, bias=False, affine=affine, track_running_stats=track_running_stats),
    "MBK7E6SE": lambda in_channels, out_channels, stride, affine, track_running_stats: InvertedResidual(in_channels, out_channels, 7, stride, expand_ratio=6, act_func="relu6", use_se=True, bias=False, affine=affine, track_running_stats=track_running_stats),
    "SkipConnect": lambda in_channels, out_channels, stride, affine, track_running_stats: SkipConnect(in_channels, out_channels, stride, act_func="relu6", bias=False, affine=affine, track_running_stats=track_running_stats)
}

class HardSwish(nn.Module):
    """hardswish activation func (see MobileNetV3)"""
    def __init__(self):
        super(HardSwish, self).__init__()
        
    def forward(self, x):
        return x * nn.ReLU6(inplace=True)(x + 3.) / 6.

class HardSigmoid(nn.Module):
    """hardsigmoid activation func used in squeeze-and-excitation module (see MobileNetV3)"""
    def __init__(self):
        super(HardSigmoid, self).__init__()
    
    def forward(self, x):
        return nn.ReLU6(inplace=True)(x + 3.) / 6.
    
class Activation(nn.Module):
    """activation function zoo"""
    def __init__(self, act_name):
        super(Activation, self).__init__()
        self.act_name = act_name
        if act_name == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act_name == "relu6":
            self.act = nn.ReLU6(inplace=True)
        elif act_name == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_name == "hardswish":
            self.act = HardSwish()
        elif act_name == "hardsigmoid":
            self.act = HardSigmoid()
        else:
            raise NotImplementedError
    
    def forward(self, x):
        return self.act(x)
    
class SEModule(nn.Module):
    """squeeze-and-excitation module implemented in linear"""
    def __init__(self, in_channels, reduction=4, act_funcs=("relu", "sigmoid"), bias=False):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction, bias=bias),
            Activation(act_funcs[0]),
            nn.Linear(in_channels//reduction, in_channels, bias=bias),
            Activation(act_funcs[1]),
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class SEModule(nn.Module):
    """squeeze-and-excitation module implemented in conv"""
    def __init__(self, in_channels, reduction=4, act_funcs=("relu", "sigmoid"), bias=False):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, stride=1, padding=0, bias=bias),
            # nn.BatchNorm2d(in_channels//reduction),
            Activation(act_funcs[0]),
            nn.Conv2d(in_channels//reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            # nn.BatchNorm2d(in_channels)
            Activation(act_funcs[1])
        )
        
    def forward(self, x):
        return x * self.se(x)
    
class SkipConnect(nn.Module):
    """skip-connect block"""
    def __init__(self, in_channels, out_channels, stride, act_func="relu6", bias=False,
                 affine=True, track_running_stats=True):
        super(SkipConnect, self).__init__()
        self.stride = stride
        self.op = None
        if in_channels != out_channels or stride != 1:
            self.op = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias),
                nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=track_running_stats),
                Activation(act_func)
            )
        
    def forward(self, x):
        return self.op(x) if self.op is not None else x
    
class InvertedResidual(nn.Module):
    """inverted residual block (see MobileNetV2)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio=3, act_func="relu6",
                 use_se=False, bias=False, affine=True, track_running_stats=True):
        super(InvertedResidual, self).__init__()
        
        assert kernel_size in [3, 5, 7]
        assert stride in [1, 2]
        assert expand_ratio <= 6
        
        padding = kernel_size // 2
        hidden_dim = round(in_channels * expand_ratio)
        self.identity = (stride == 1) and (in_channels == out_channels)
        
        self.main = []
        if expand_ratio == 1:
            self.main += [
                # dw
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias),
                nn.BatchNorm2d(in_channels, affine=affine, track_running_stats=track_running_stats),
            ]
            if use_se:
                self.main.append(SEModule(in_channels))
            self.main += [
                Activation(act_func),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
                nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=track_running_stats),
            ]
        else:
            self.main += [
                # pw
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=bias),
                nn.BatchNorm2d(hidden_dim, affine=affine, track_running_stats=track_running_stats),
                Activation(act_func),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=hidden_dim, bias=bias),
                nn.BatchNorm2d(hidden_dim, affine=affine, track_running_stats=track_running_stats),
            ]
            if use_se:
                self.main.append(SEModule(hidden_dim))
            self.main += [
                Activation(act_func),
                # pw-linear
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
                nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=track_running_stats),
            ]
            
        self.main = nn.Sequential(*self.main)
    
    def forward(self, x):
        if self.identity:
            return x + self.main(x)
        else:
            return self.main(x)
