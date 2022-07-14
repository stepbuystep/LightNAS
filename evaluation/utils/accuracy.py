# -*- coding: utf-8 -*-
from __future__ import print_function, division

def accuracy(output, target, topk=(1,)):
    """
    Calculate the topk accuracy
    """
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0))
    return res
