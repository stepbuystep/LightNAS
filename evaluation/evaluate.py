# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch

from models.network import Network
from utils.dataloader import get_imagenet_dataset
from utils.accuracy import accuracy

parser = argparse.ArgumentParser(description='LightNet Config')
parser.add_argument('--model', default='LightNet-20ms', help="model name")
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help="use cpu or gpu to evaluate")
parser.add_argument('--dataset-root', default='/Your_Root/ILSVRC2012', help="dataset root path")
parser.add_argument('--batch-size', default=64, type=int, help='batch size')
parser.add_argument('--gpu-id', default=0, type=int, help='gpu id')
args = parser.parse_args()

if __name__ == "__main__":
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.model == "LightNet-20ms":
        arch_config = ['MBK3E1', 'SkipConnect', 'MBK3E3', 'MBK3E6', 'MBK7E3', 'MBK3E3', 'MBK5E3', 'MBK7E3', 'MBK3E6', 'MBK3E3', 'MBK7E6', 'MBK5E3', 'MBK7E6', 'MBK7E3', 'MBK5E3', 'MBK7E3', 'SkipConnect', 'MBK5E3', 'MBK7E3', 'MBK7E3', 'MBK3E6', 'MBK7E6']
    elif args.model == "LightNet-22ms":
        arch_config = ['MBK3E1', 'MBK7E6', 'SkipConnect', 'MBK7E6', 'MBK3E6', 'SkipConnect', 'SkipConnect', 'MBK5E3', 'MBK7E6', 'MBK7E6', 'MBK7E3', 'MBK7E3', 'SkipConnect', 'MBK7E3', 'MBK7E6', 'MBK7E6', 'SkipConnect', 'MBK5E6', 'MBK7E3', 'MBK7E3', 'MBK7E3', 'MBK5E6']
    elif args.model == "LightNet-24ms":
        arch_config = ['MBK3E1', 'MBK3E3', 'MBK7E6', 'MBK3E3', 'MBK5E6', 'MBK5E6', 'MBK7E3', 'MBK3E6', 'SkipConnect', 'MBK7E6', 'MBK3E6', 'MBK3E3', 'MBK7E3', 'MBK5E3', 'MBK7E3', 'SkipConnect', 'SkipConnect', 'MBK7E3', 'MBK7E3', 'MBK5E3', 'MBK5E6', 'MBK3E6']
    elif args.model == "LightNet-26ms":
        arch_config = ['MBK3E1', 'SkipConnect', 'MBK7E6', 'MBK7E6', 'MBK3E6', 'MBK7E6', 'MBK3E3', 'MBK5E3', 'MBK5E6', 'MBK7E6', 'MBK3E6', 'SkipConnect', 'MBK7E6', 'MBK7E6', 'MBK7E6', 'MBK3E3', 'MBK5E6', 'MBK5E3', 'SkipConnect', 'MBK5E6', 'MBK7E3', 'MBK5E6']
    elif args.model == "LightNet-28ms":
        arch_config = ['MBK3E1', 'MBK5E3', 'MBK7E6', 'MBK3E3', 'MBK7E6', 'MBK5E6', 'MBK5E6', 'MBK5E3', 'MBK5E3', 'MBK3E6', 'MBK7E3', 'MBK3E3', 'MBK7E6', 'MBK3E3', 'MBK7E6', 'MBK7E3', 'SkipConnect', 'MBK5E6', 'MBK5E6', 'MBK7E6', 'MBK5E6', 'MBK7E6']
    elif args.model == "LightNet-30ms":
        arch_config = ['MBK3E1', 'MBK3E6', 'MBK7E3', 'MBK3E6', 'MBK7E6', 'MBK7E6', 'MBK3E3', 'MBK3E6', 'MBK3E3', 'MBK7E6', 'MBK5E6', 'MBK7E6', 'MBK7E6', 'MBK7E3', 'SkipConnect', 'MBK7E6', 'MBK7E6', 'MBK5E6', 'SkipConnect', 'MBK7E6', 'MBK7E6', 'MBK3E6']
    elif args.model == "LightNet-20ms-SE":
        arch_config = ['MBK3E1', 'SkipConnect', 'MBK3E3', 'MBK3E6', 'MBK7E3', 'MBK3E3', 'MBK5E3', 'MBK7E3', 'MBK3E6', 'MBK3E3', 'MBK7E6', 'MBK5E3', 'MBK7E6', 'MBK7E3SE', 'MBK5E3SE', 'MBK7E3SE', 'SkipConnect', 'MBK5E3SE', 'MBK7E3SE', 'MBK7E3SE', 'MBK3E6SE', 'MBK7E6SE']
    elif args.model == "LightNet-22ms-SE":
        arch_config = ['MBK3E1', 'MBK7E6', 'SkipConnect', 'MBK7E6', 'MBK3E6', 'SkipConnect', 'SkipConnect', 'MBK5E3', 'MBK7E6', 'MBK7E6', 'MBK7E3', 'MBK7E3', 'SkipConnect', 'MBK7E3SE', 'MBK7E6SE', 'MBK7E6SE', 'SkipConnect', 'MBK5E6SE', 'MBK7E3SE', 'MBK7E3SE', 'MBK7E3SE', 'MBK5E6SE']
    elif args.model == "LightNet-24ms-SE":
        arch_config = ['MBK3E1', 'MBK3E3', 'MBK7E6', 'MBK3E3', 'MBK5E6', 'MBK5E6', 'MBK7E3', 'MBK3E6', 'SkipConnect', 'MBK7E6', 'MBK3E6', 'MBK3E3', 'MBK7E3', 'MBK5E3SE', 'MBK7E3SE', 'SkipConnect', 'SkipConnect', 'MBK7E3SE', 'MBK7E3SE', 'MBK5E3SE', 'MBK5E6SE', 'MBK3E6SE']
    elif args.model == "LightNet-26ms-SE":
        arch_config = ['MBK3E1', 'SkipConnect', 'MBK7E6', 'MBK7E6', 'MBK3E6', 'MBK7E6', 'MBK3E3', 'MBK5E3', 'MBK5E6', 'MBK7E6', 'MBK3E6', 'SkipConnect', 'MBK7E6', 'MBK7E6SE', 'MBK7E6SE', 'MBK3E3SE', 'MBK5E6SE', 'MBK5E3SE', 'SkipConnect', 'MBK5E6SE', 'MBK7E3SE', 'MBK5E6SE']
    elif args.model == "LightNet-28ms-SE":
        arch_config = ['MBK3E1', 'MBK5E3', 'MBK7E6', 'MBK3E3', 'MBK7E6', 'MBK5E6', 'MBK5E6', 'MBK5E3', 'MBK5E3', 'MBK3E6', 'MBK7E3', 'MBK3E3', 'MBK7E6', 'MBK3E3SE', 'MBK7E6SE', 'MBK7E3SE', 'SkipConnect', 'MBK5E6SE', 'MBK5E6SE', 'MBK7E6SE', 'MBK5E6SE', 'MBK7E6SE']
    elif args.model == "LightNet-30ms-SE":
        arch_config = ['MBK3E1', 'MBK3E6', 'MBK7E3', 'MBK3E6', 'MBK7E6', 'MBK7E6', 'MBK3E3', 'MBK3E6', 'MBK3E3', 'MBK7E6', 'MBK5E6', 'MBK7E6', 'MBK7E6', 'MBK7E3SE', 'SkipConnect', 'MBK7E6SE', 'MBK7E6SE', 'MBK5E6SE', 'SkipConnect', 'MBK7E6SE', 'MBK7E6SE', 'MBK3E6SE']
    else:
        raise NotImplementedError

    model = Network(arch_config=arch_config)
    device = torch.device(args.device)
    pretrained_path = os.path.join("./pretrained", args.model+".pth.tar")
    state_dict = torch.load(pretrained_path, map_location=device)["state_dict"]
    # process state_dict
    state_dict_curr = {}
    for name in state_dict:
        name_curr = name[7:]
        state_dict_curr[name_curr] = state_dict[name]
    model.load_state_dict(state_dict_curr)
    if device.type == 'cuda':
        model.cuda()
    model.eval()

    val_dataloader = get_imagenet_dataset(batch_size=args.batch_size,
                                          dataset_root=args.dataset_root,
                                          dataset_tpye="valid")

    print("Start to evaluate <{}> ...".format(args.model))
    total_top1 = 0.0
    total_top5 = 0.0
    total_counter = 0.0
    for image, label in val_dataloader:
        image, label = image.to(device), label.to(device)
        with torch.no_grad():
            result = model(image)
            top1, top5 = accuracy(result, label, topk=(1, 5))
        if device.type == 'cuda':
            total_counter += image.cpu().data.shape[0]
            total_top1 += top1.cpu().data.numpy()
            total_top5 += top5.cpu().data.numpy()
        else:
            total_counter += image.data.shape[0]
            total_top1 += top1.data.numpy()
            total_top5 += top5.data.numpy()
    mean_top1 = total_top1 / total_counter
    mean_top5 = total_top5 / total_counter
    print('Evaluate Result: Total: %d\tTop1-Acc: %.3f\tTop5-Acc: %.3f' % (total_counter, mean_top1, mean_top5))
