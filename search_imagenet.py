# -*- coding: utf-8 -*-
import os
import sys
import argparse
import random
import shutil
import time
import glob
import logging
import math

import numpy as np
import scipy.io as io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.autograd import Variable

import utils
from models import SuperNetwork, ops_all

import warnings
warnings.filterwarnings("ignore")

EXP_PATH = "./experiments/"
global logger

def parse_args():
    parser = argparse.ArgumentParser(description="ImageNet Training")
    parser.add_argument("--seed", type=int, default=10, help="seed")
    parser.add_argument("--save", type=str, default="/path/to/exp", help="save experiments")
    parser.add_argument("--dataset", type=str, default="/home/user/Downloads/imagenet100", help="path to dataset")
    parser.add_argument("--epochs", type=int, default=90,  help="number of training epochs")
    parser.add_argument("--freeze_epochs", type=int, default=10, help="number of freeze epochs to update the network weights only")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="number of warmup epochs")
    parser.add_argument("--num_workers", type=int, default=16,  help="number of workers for data loading")
    parser.add_argument("--batch_size_train", type=int, default=128, help="batch size for train_datatset")
    parser.add_argument("--batch_size_val", type=int, default=128, help="batch size for val_dataset")
    parser.add_argument("--target_latency", type=float, default=20, help="target latency constraint")
    #
    parser.add_argument("--model_learning_rate", type=float, default=0.1, help="model weights learning rate")
    parser.add_argument("--tradeoff", type=float, default=0., help="trade-off factor between accuracy and latency")
    parser.add_argument("--tradeoff_trunc", action="store_true", default=True, help="whether to trunc tradeoff at 0.")
    parser.add_argument("--tradeoff_learning_rate", type=float, default=0.0005, help="learning rate of trade-off")
    parser.add_argument("--model_weight_decay", type=float, default=3e-5, help="model weight decay in SGD")
    parser.add_argument("--model_momentum", type=float, default=0.9, help="momentum in SGD")
    parser.add_argument("--alphas_learning_rate", type=float, default=0.001, help="alphas weights learning rate")
    parser.add_argument("--alphas_weight_decay", type=float, default=1e-3, help="alphas weight decay in Adam")
    parser.add_argument("--grad_clip", type=float, default=5., help="gradient clipping")
    parser.add_argument("--label_smooth", type=float, default=0.1, help="label smoothing")
    parser.add_argument("--gpus", type=int, default=1, help="number of GPUs for training (default: 4)")
    parser.add_argument("--visible", type=str, default="0", help="environment visible GPUs")
    parser.add_argument("--nodes", type=int, default=1, help="number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="node rank for current process")
    parser.add_argument("--print_freq", type=int, default=50, help="log print frequency")
    parser.add_argument("--syncbn", action="store_true", default=True, help="synchronize bn across multiple gpus")
    parser.add_argument("--resume", type=str, default=None, help="resume training and path to checkpoints should be specified")
    parser.add_argument("--init_method", type=str, default="tcp://155.69.146.202:22717", help="e.g., tcp://12.345.6.6:3282")
    # network setting
    parser.add_argument("--width_mult", type=float, default=1., help="width multiplifier")
    parser.add_argument("--num_classes", type=int, default=100, help="number of classes")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate in last conv")
    parser.add_argument("--channel_layout", type=str, default="mobilenetv2", help="mobilenetv2 channel layout")
    parser.add_argument("--affine", action="store_true", default=False, help="affine flag in batchnorm")
    parser.add_argument("--track_running_stats", action="store_true", default=True, help="track_running_stats flag in batchnorm")
    parser.add_argument("--groups", type=int, default=1, help="number of groups for partial channels")
    parser.add_argument("--tau_max", type=float, default=5.0, help="maximum of tau in gumbel-softmax")
    parser.add_argument("--tau_min", type=float, default=0.001, help="minimum of tau in gumbel-softmax")
    parser.add_argument("--eta", type=float, default=0., help="eta in identity mapping")
    args, unparsed = parser.parse_known_args()
    return args

args = parse_args()
args.save = EXP_PATH + args.save
utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*.py"))
logger_path = os.path.join(args.save, "log.txt")
logger = utils.set_logger(logger_path)
op_names = ["MBK3E3", "MBK3E6", "MBK5E3", "MBK5E6", "MBK7E3", "MBK7E6", "SkipConnect"]

def main():
    """main func"""

    logger.info("args = %s"%args)
    if not torch.cuda.is_available():
        logger.info("something wrong here")
        sys.exit(1)
    assert len(args.visible.split(",")) == args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # enable cudnn optimization
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    args.world_size = args.gpus * args.nodes
    mp.spawn(main_worker, nprocs=args.gpus, args=(args,))


def main_worker(gpu, args):
    """single process for distributed training"""

    args.gpu = gpu
    rank = args.node_rank * args.gpus + gpu
    dist.init_process_group(
        backend="nccl",
        init_method=args.init_method,
        world_size=args.world_size,
        rank=rank
    )
    # set default gpu device in current process
    logger.info("Use GPU: {} for training".format(args.gpu))
    torch.cuda.set_device(args.gpu)

    model = SuperNetwork(width_mult=args.width_mult, dropout=args.dropout, num_classes=args.num_classes,
                         channel_layout=args.channel_layout, affine=args.affine, track_running_stats=args.track_running_stats, op_names=op_names, groups=args.groups).cuda(args.gpu)
    model.init_alphas(args.gpu)
    # print(model.get_alphas())

    # resume weight except for the network weights
    if args.resume is not None:
        if os.path.exists(args.resume):
            resume = os.path.join(args.resume, "best.pth.tar")
            checkpoint = torch.load(resume, map_location="cuda:{}".format(args.gpu))
            model.alphas = Variable(checkpoint["alphas"].cuda(args.gpu), requires_grad=True)

    if args.syncbn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    model_optimizer = optim.SGD(model.parameters(), lr=args.model_learning_rate, momentum=args.model_momentum, weight_decay=args.model_weight_decay)
    model_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, args.epochs)
    alphas_optimizer = optim.Adam([model.module.get_alphas()], lr=args.alphas_learning_rate, betas=(0.5, 0.999), weight_decay=args.alphas_weight_decay)
    # alphas_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(alphas_optimizer, args.epochs)

    if args.label_smooth > 0.:
        criterion = CrossEntropyLabelSmooth(args.label_smooth, num_classes=args.num_classes).cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # resume network weights
    args.start_epoch = 0
    args.end_epoch = args.epochs
    if args.resume is not None:
        if os.path.exists(args.resume):
            args.resume = os.path.join(args.resume, "best.pth.tar")
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cuda:{}".format(args.gpu))
            model.load_state_dict(checkpoint["state_dict"])
            model_optimizer.load_state_dict(checkpoint["model_optimizer"])
            model_lr_scheduler.load_state_dict(checkpoint["model_lr_scheduler"])
            # model.module.arch_weights = checkpoint["arch_params"]
            alphas_optimizer.load_state_dict(checkpoint["alphas_optimizer"])
            args.start_epoch = checkpoint["epoch"]
            alphas_optimizer.param_groups[0]["lr"] = args.alphas_learning_rate
            logger.info("=> loaded checkpoint '{}' epoch {}"
                       .format(args.resume, args.start_epoch))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # data loading
    train_dir = os.path.join(args.dataset, "train")
    val_dir = os.path.join(args.dataset, "val")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            contrast=0.4,
            brightness=0.4,
            saturation=0.4,
            hue=0.1
        ),
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    val_dataset = datasets.ImageFolder(val_dir, val_transform)
    # set distributed sampler, thus shuffle option in train_loader should be False
    if args.gpus > 1:
        args.batch_size_train = int(args.batch_size_train / args.gpus)
        args.batch_size_val = int(args.batch_size_val / args.gpus)
        args.num_workers = int(args.num_workers / args.gpus)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_train,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        sampler=val_sampler
    )

    # args.tradeoff = Variable(torch.tensor(args.tradeoff).cuda(args.gpu), requires_grad=True)
    args.count = 0
    for epoch in range(args.start_epoch, args.end_epoch):
        # args.tau_curr = args.tau_max - (args.tau_max - args.tau_min) * (epoch - args.freeze_epochs) / (args.end_epoch - 1 - args.freeze_epochs)  if epoch >= args.freeze_epochs else args.tau_max
        
        args.tau_curr = args.tau_max * math.exp(-0.1 * (epoch - args.freeze_epochs)) if epoch >= args.freeze_epochs else args.tau_max
        args.eta_curr = args.eta - args.eta * (epoch - args.freeze_epochs) / (args.end_epoch - 1 - args.freeze_epochs) if epoch >= args.freeze_epochs else args.eta
        # learning rate warmup
        if epoch < args.warmup_epochs:
            model_optimizer.param_groups[0]["lr"] = args.model_learning_rate * (epoch + 1) / args.warmup_epochs
        
        if epoch >= args.freeze_epochs:
            args.alphas_learning_rate_curr = args.alphas_learning_rate * (0.5 ** ((epoch - args.freeze_epochs)//20))
            if args.epochs - (epoch+1) >= 10:
                alphas_optimizer.param_groups[0]["lr"] = args.alphas_learning_rate_curr

        if args.gpu == 0:
            logger.info("Epoich: {} / {}, MODEL_LR: {}, ALPHAS_LR: {}, TAU: {}, ETA: {}".format(epoch+1, args.end_epoch,
                                                                            model_optimizer.param_groups[0]["lr"],
                                                                            alphas_optimizer.param_groups[0]["lr"], args.tau_curr, args.eta_curr))

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_epoch(train_loader, model, criterion, model_optimizer, epoch, args, mode="Train")
        if epoch >= args.freeze_epochs:
            train_epoch(val_loader, model, criterion, alphas_optimizer, epoch, args, mode="Valid")

        model_lr_scheduler.step()
        # alphas_lr_scheduler.step()

        if epoch == args.freeze_epochs - 1:
            flag = True
        else:
            flag = False
        if args.gpu == 0:
            utils.save_checkpoint({
                "epoch": epoch+1,
                "state_dict": model.state_dict(),
                "model_optimizer": model_optimizer.state_dict(),
                "model_lr_scheduler": model_lr_scheduler.state_dict(),
                "alphas_optimizer": alphas_optimizer.state_dict(),
                "alphas": model.module.get_alphas()
            }, args.save, flag)

def train_epoch(dataloader, model, criterion, optimizer, epoch, args, mode="Train"):
    """train for one epoch"""

    predictor = Predictor().cuda(args.gpu)
    predictor.load_state_dict(torch.load("./data/latency_predictor.pth.tar", map_location="cuda:{}".format(args.gpu)))

    batch_time = utils.AverageMeter("Time")
    data_time = utils.AverageMeter("Data")
    losses = utils.AverageMeter("Loss")
    top1 = utils.AverageMeter("Acc@1")
    top5 = utils.AverageMeter("Acc@5")
    total_step = len(dataloader)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, targets) in enumerate(dataloader):
        # move to default cuda
        inputs = inputs.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
        data_time.update(time.time() - end)

        if epoch >= args.freeze_epochs:
            optimizer.zero_grad()

            alphas_gumbels = utils.gumbel_softmax(model.module.get_alphas(), tau=args.tau_curr, hard=True)    
            outputs = model(inputs, alphas_gumbels=alphas_gumbels, eta=args.eta_curr)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            if args.tradeoff is not None and mode == "Valid":
                latency_curr = predictor(alphas_gumbels.view(1, -1)).reshape(-1)
                latency_loss = args.tradeoff * (latency_curr / args.target_latency - 1)
                latency_loss.backward(retain_graph=True)

            loss.backward()

            losses.update(loss.item(), inputs.size()[0])
            top1.update(acc1.item(), inputs.size()[0])
            top5.update(acc5.item(), inputs.size()[0])

            if args.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if args.tradeoff is not None and mode == "Valid":
                latency_curr = predictor(utils.arch_to_onehot(model.module.parse_alphas()).cuda(args.gpu).view(1, -1))
                args.tradeoff += args.tradeoff_learning_rate * (latency_curr / args.target_latency - 1)
                args.tradeoff = args.tradeoff.item()
                if args.epochs - (epoch+1) < 10:
                    if abs(latency_curr.item() - args.target_latency) < 0.15:
                        args.count = 1
                    if args.count == 1:
                        optimizer.param_groups[0]["lr"] = args.alphas_learning_rate_curr * abs(latency_curr.item() / args.target_latency - 1) # * (args.epochs - epoch) / 5.
                    else:
                        pass
            if args.tradeoff is not None and args.tradeoff_trunc and mode == "Valid":
                if latency_curr > args.target_latency and args.tradeoff < 0:
                    args.tradeoff = args.tradeoff * 0.
                if latency_curr < args.target_latency and args.tradeoff > 0:
                    args.tradeoff = args.tradeoff * 0.

            batch_time.update(time.time() - end)
            end = time.time()
        else:
            num_paths = len(op_names)
            alphas_gumbels_list = utils.gumbel_softmax_topk(model.module.get_alphas(), tau=args.tau_curr, hard=True, k=len(op_names))
            optimizer.zero_grad()
            for path_idx, alphas_gumbels in enumerate(alphas_gumbels_list):
                outputs = model(inputs, alphas_gumbels=alphas_gumbels, eta=args.eta_curr)
                loss = criterion(outputs, targets)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

                loss.backward(retain_graph=True)

                losses.update(loss.item(), inputs.size()[0])
                top1.update(acc1.item(), inputs.size()[0])
                top5.update(acc5.item(), inputs.size()[0])

            if args.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

        if i != 0 and i % args.print_freq == 0 and args.gpu == 0:
            logger.info("[TRAIN EPOCH ({mode}): {epoch:<3d}] * Process ID: {gpu}  [{step:<4d} / {total_step}]  Loss: {losses.avg:.4e}, "
                            "Batch: {batch_time.sum:.1f}s, "
                            "Data: {data_time.sum:.1f}s, "
                            "Acc@1: {top1.avg:.3f}%, "
                            "Acc@5: {top5.avg:.3f}%,  "
                            "Tradeoff: {tradeoff}"
                            .format(mode=mode, epoch=epoch+1, end_epoch=args.end_epoch, gpu=args.gpu, step=i, total_step=total_step, losses=losses,
                                   batch_time=batch_time, data_time=data_time,
                                   top1=top1, top5=top5, tradeoff=args.tradeoff))

    if args.gpu == 0:
        logger.info("[TRAIN EPOCH ({mode}): {epoch:<3d}] * Process ID: {gpu}  Loss: {losses.avg:.4e}, "
                    "Time (Loading / Training): {data_time.sum:.1f}s / {batch_time.sum:.1f}s, "
                    "Acc@1: {top1.avg:.3f}%, "
                    "Acc@5: {top5.avg:.3f}%, "
                    "Tradeoff: {tradeoff}"
                    .format(mode=mode, epoch=epoch+1, gpu=args.gpu, losses=losses, data_time=data_time,
                                                   batch_time=batch_time, top1=top1, top5=top5, tradeoff=args.tradeoff))
        latency_mlp = predictor(utils.arch_to_onehot(model.module.parse_alphas()).cuda(args.gpu).view(1, -1))
        logger.info("Architecture: {}, Latency: {}ms".format(model.module.parse_alphas(), latency_mlp.item()))

    return top1.avg, top5.avg


class CrossEntropyLabelSmooth(nn.Module):
    """cross entropy with label smooth"""

    def __init__(self, epsilon, num_classes=1000):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, outputs, targets):
        log_probs = self.log_softmax(outputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

def accuracy(outputs, targets, topk=(1, 5)):
    """compute the accuracy over topk predictions"""

    with torch.no_grad():
        batch_size = outputs.size()[0]
        maxk = max(topk)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append(correct_k.mul_(100. / batch_size))

    return results

class Predictor(nn.Module):
    """latency predictor"""
    def __init__(self):
        super(Predictor, self).__init__()
        self.pred = nn.Sequential(
            nn.Linear(147, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1, bias=True)
        )

    def forward(self, x):
        return self.pred(x)

if __name__ == "__main__":
    main()
