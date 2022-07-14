# -*- coding: utf-8 -*-
import torch
import time
import logging
import os
import shutil
import random
import numpy as np

def set_logger(path):
    """set up logging txt"""
    logger = logging.getLogger("imagenet-training")
    log_format = "%(asctime)s | %(message)s"
    formater  = logging.Formatter(log_format, datefmt="%m/%d %I:%M:%S %p")
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formater)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formater)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

class AverageMeter(object):
    """average meter"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.value = 0.
        self.count = 0.
        self.sum = 0.
        self.avg = 0.

    def update(self, value, n=1):
        self.value = value
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count

def count_parameters_in_MB(model):
    return np.sum(np.prod(param.size()) for name, param in model.named_parameters() if "aux" not in name) / 1e6

def create_exp_dir(path, scripts_to_save=None):
    """set up experiment env"""
    if not os.path.exists(path):
        os.mkdir(path)

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, "scripts")):
            os.mkdir(os.path.join(path, "scripts"))
        for script in scripts_to_save:
            dst_file = os.path.join(path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)

def save_checkpoint(state_dict, path, is_best=False):
    """save checkpoints"""
    filename = os.path.join(path, "checkpoint.pth.tar")
    torch.save(state_dict, filename)
    if is_best:
        best_filename = os.path.join(path, "best.pth.tar")
        shutil.copyfile(filename, best_filename)

def measure_latency(model, batch_size=1, N=20, use_cuda=False):
    """latency measurements"""
    if use_cuda:
        assert torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.enabled

    inputs = torch.randn(batch_size, 3, 224, 224)
    model = model.eval()
    if use_cuda:
        inputs = inputs.cuda()
        model = model.cuda()

    latency_list = []
    with torch.no_grad():
        # warm up
        for i in range(5):
            _ = model(inputs)
        for i in range(N):
            start = time.time()
            _ = model(inputs)
            if use_cuda:
                torch.cuda.synchronize()
            latency_list.append(time.time() - start)
    latency_list.sort()

    return sum(latency_list[:int(N*2/3)]) / int(N*2/3) * 1000 # ms

def arch_to_onehot(arch_config, op_names=None):
    """convert arch_config to one-hot matrix"""

    if op_names is None:
        op_names = ["MBK3E3", "MBK3E6", "MBK5E3", "MBK5E6", "MBK7E3", "MBK7E6", "SkipConnect"]

    assert arch_config[0] == "MBK3E1"

    index_list = []
    for block in arch_config[1:]:
        index_list.append(op_names.index(block))

    index_np = np.asarray(index_list)
    index_pt = torch.from_numpy(index_np)

    results = torch.zeros(len(arch_config)-1, len(op_names)).scatter_(1, index_pt.unsqueeze_(1), 1.)
    return results

def matrix_to_weight(data, op_names=None, channel_layout="proxylessnas"):
    """convert lut matrix to weight"""

    if op_names is None:
        op_names = ["MBK3E3", "MBK3E6", "MBK5E3", "MBK5E6", "MBK7E3", "MBK7E6", "SkipConnect"]

    first_conv = data["first_conv"][0][0]
    first_block = data["0_MBK3E1_1"][0][0] # MBK3E1
    last_conv = data["last_conv"][0][0]
    classifier = data["classifier"][0][0]
    others = first_conv + first_block + last_conv + classifier

    results = torch.zeros(21, len(op_names))
    for i in range(1, 22):
        for j, op_name in enumerate(op_names):
            if channel_layout == "proxylessnas":
                if i in [1, 5, 9, 17]:
                    stride = 2
                else:
                    stride = 1
            elif channel_layout == "densenas":
                if i in [1, 3, 6, 13]:
                    stride = 2
                else:
                    stride = 1
            else:
                raise ValueError("something wrong here")
            block_name = "{}_{}_{}".format(i, op_name, stride)
            results[i-1][j] = data[block_name][0][0]

    return results, others

def gumbel_softmax(logits, tau=1., hard=False, dim=-1):
    """gumbel-softmax reparameterization"""

    logits_log_probs = torch.log(torch.softmax(logits, dim=dim))
    gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
    # gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits_log_probs + gumbels) / tau
    logits_soft = torch.softmax(gumbels, dim=dim)

    if hard:
        index = logits_soft.max(dim, keepdim=True)[1]
        logits_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
        results = logits_hard - logits_soft.detach() + logits_soft
    else:
        results = logits_soft

    return results

def gumbel_softmax_topk(logits, tau=1., hard=False, dim=-1, k=5):
    """Gumbel-Softmax Top-K Reparameterization"""

    logits_probs = torch.softmax(logits, dim=dim)
    results = []
    count = 0

    while count < k:
        logits_log_probs = torch.log(logits_probs + 1e-20) # avoid log(0)
        gumbels = -torch.log(-torch.log(torch.rand_like(logits, memory_format=torch.legacy_contiguous_format)))
        # gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        gumbels = (logits_log_probs + gumbels) / tau
        y_soft = torch.softmax(gumbels, dim=dim)
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter(dim, index, 1.)

        if hard:
            results.append(y_hard - y_soft.detach() + y_soft)
        else:
            results.append(y_soft)
        count += 1
        y_hard = 1 - y_hard.clone().detach() # without shared memory and without autograd
        logits_probs = (y_hard * logits_probs) / (y_hard * logits_probs).sum(dim=dim, keepdim=True)

    return results
