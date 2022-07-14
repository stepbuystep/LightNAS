# -*- coding: utf-8 -*-
import os
import sys
import argparse
import random
import shutil
import time
import glob
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets

from thop import profile
from torchtoolbox.tools import no_decay_bias

import utils
from models import Network

import warnings
warnings.filterwarnings("ignore")

is_best = False
best_acc1 = 0.
best_acc5 = 0.
best_epoch = 0
EXP_PATH = "./experiments/"
global logger

def parse_args():
    parser = argparse.ArgumentParser(description="ImageNet Training")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--save", type=str, default="/path/to/exp", help="save experiments")
    parser.add_argument("--dataset", type=str, default="/home/getluo/Download/imagenet", help="path to dataset")
    parser.add_argument("--epochs", type=int, default=360,  help="number of training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="number of warmup epochs")
    parser.add_argument("--num_workers", type=int, default=32,  help="number of workers for data loading")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=4e-5, help="weight decay in SGD")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum in SGD")
    parser.add_argument("--grad_clip", type=float, default=5., help="gradient clipping")
    parser.add_argument("--label_smooth", type=float, default=0.1, help="label smoothing")
    parser.add_argument("--gpus", type=int, default=4, help="number of GPUs for training (default: 4)")
    parser.add_argument("--visible", type=str, default="0,1,2,3", help="environment visible GPUs")
    parser.add_argument("--print_freq", type=int, default=100, help="log print frequency")
    parser.add_argument("--evaluate", action="store_true", default=False, help="just evaluate or not")
    parser.add_argument("--resume", type=str, default=None, help="resume training and path to checkpoints should be specified")
    parser.add_argument("--no_weight_decay_bias", action="store_true", default=True, help="remove weight decay for bias and bn")
    # network setting
    parser.add_argument("--arch_config", type=str, default="", help="arch configurations")
    parser.add_argument("--width_mult", type=float, default=1., help="width multiplifier")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate in last conv")
    parser.add_argument("--channel_layout", type=str, default="mobilenetv2", help="mobilenetv2 channel layout")
    parser.add_argument("--num_classes", type=int, default=1000, help="number of classes")
    args, unparsed = parser.parse_known_args()
    return args

args = parse_args()
args.arch_config = eval(args.arch_config)
args.save = EXP_PATH + args.save
utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*.py"))
logger_name = os.path.join(args.save, "log.txt")
logger = utils.set_logger(logger_name)

def main():
    """"""

    global best_acc1, best_acc5, is_best, best_epoch

    logger.info("args = %s"%args)
    if not torch.cuda.is_available():
        logger.info("No GPUs available!")
        sys.exit(1)
    assert len(args.visible.split(",")) == args.gpus
    args.visible = list(map(int, args.visible.split(",")))
    torch.cuda.set_device("cuda:{}".format(args.visible[0]))

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # enable cudnn optimization
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model = Network(arch_config=args.arch_config, width_mult=args.width_mult, dropout=args.dropout,
                    channel_layout=args.channel_layout, num_classes=args.num_classes).cuda()
    # profile flops and params
    inputs = torch.randn(1, 3, 224, 224).cuda()
    flops, params = profile(model, inputs=(inputs,))
    logger.info("FLOPs: {}, Params: {}".format(flops, params))
    model = nn.DataParallel(model, device_ids=args.visible).cuda()

    if args.no_weight_decay_bias:
        model_params = no_decay_bias(model)
    else:
        model_params = model.parameters()
    optimizer = optim.SGD(model_params, lr=args.learning_rate,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.label_smooth is not None:
        criterion = CrossEntropyLabelSmooth(epsilon=args.label_smooth, num_classes=args.num_classes).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    # resume
    args.start_epoch = 0
    args.end_epoch = args.epochs
    if args.resume is not None:
        if os.path.exists(args.resume):
            args.resume = os.path.join(args.resume, "best.pth.tar")
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cuda:{}".format(args.visible[0]))
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            best_acc5 = checkpoint["best_acc5"]
            logger.info("=> loaded checkpoint '{}' epoch {}"
                       .format(args.resume, args.start_epoch))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # data loading code
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
    train_dataset = datasets.ImageFolder(
        train_dir,
        train_transform)
    val_dataset = datasets.ImageFolder(
        val_dir,
        val_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.end_epoch):

        # learning rate warmup
        if epoch < args.warmup_epochs:
            optimizer.param_groups[0]["lr"] = args.learning_rate * (epoch + 1) / args.warmup_epochs

        logger.info("Epoich: {} / {}, LR: {}".format(epoch+1, args.end_epoch, optimizer.param_groups[0]["lr"]))

        # train for one epoch
        train_epoch(train_loader, model, criterion, optimizer, epoch, args)

        # decay learning rate
        lr_scheduler.step()

        # evaluate after training
        acc1, acc5 = validate(val_loader, model, criterion, args)

        if acc1 > best_acc1:
            is_best = True
            best_acc1 = acc1
            best_acc5 = acc5
            best_epoch = epoch + 1
        else:
            is_best = False

        utils.save_checkpoint({
            "epoch": epoch+1,
            "state_dict": model.state_dict(),
            "best_acc1": acc1,
            "best_acc5": acc5,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
        }, args.save, is_best)

        logger.info("Best Epoch: {}, Best Acc@1: {:.3f}, Acc@5: {:.3f}".format(best_epoch, best_acc1, best_acc5))

    logger.info("Best Acc@1: {:.3f}, Acc@5: {:.3f}".format(best_acc1, best_acc5))

class CrossEntropyLabelSmooth(nn.Module):
    """Label Smoothing"""
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

def accuracy(outputs, targets, topk=(1,)):
    """Compute the accuracy over the k top preds"""
    with torch.no_grad():
        batch_size = outputs.size()[0]
        maxk = max(topk)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res.append(correct_k.mul_(100. / batch_size))

    return res

def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
    """train for one epoch"""

    batch_time = utils.AverageMeter("Time")
    data_time = utils.AverageMeter("Data")
    losses = utils.AverageMeter("Loss")
    top1 = utils.AverageMeter("Acc@1")
    top5 = utils.AverageMeter("Acc@5")
    total_step = len(train_loader)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):

        # move to default cuda
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - end)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # update meters
        losses.update(loss.item(), inputs.size()[0])
        top1.update(acc1.item(), inputs.size()[0])
        top5.update(acc5.item(), inputs.size()[0])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i != 0 and i % args.print_freq == 0:
            logger.info("[TRAIN EPOCH: {epoch:<3d}], [{step:<4d} / {total_step}]  Loss: {losses.avg:.4e}, "
                            "Batch: {batch_time.sum:.1f}s, "
                            "Data: {data_time.sum:.1f}s, "
                            "Acc@1: {top1.avg:.3f}%, "
                            "Acc@5: {top5.avg:.3f}%  "
                            .format(epoch=epoch+1, step=i, total_step=total_step, losses=losses,
                                   batch_time=batch_time, data_time=data_time,
                                   top1=top1, top5=top5))

    logger.info("[TRAIN EPOCH: {epoch:<3d}],  Loss: {losses.avg:.4e}, "
                "Time (Loading / Training): {data_time.sum:.1f}s / {batch_time.sum:.1f}s, "
                "Acc@1: {top1.avg:.3f}%, "
                "Acc@5: {top5.avg:.3f}%".format(epoch=epoch+1, losses=losses, data_time=data_time, batch_time=batch_time, top1=top1, top5=top5))

    return top1.avg, top5.avg

def validate(val_loader, model, criterion, args):
    """evaluate trained architecture"""

    batch_time = utils.AverageMeter("Batch")
    data_time = utils.AverageMeter("Data")
    top1 = utils.AverageMeter("Acc@1")
    top5 = utils.AverageMeter("Acc@5")
    losses = utils.AverageMeter("Loss")
    total_step = len(val_loader)

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets) in enumerate(val_loader):

            # move to default cuda
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            data_time.update(time.time() - end)

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size()[0])
            top1.update(acc1.item(), inputs.size()[0])
            top5.update(acc5.item(), inputs.size()[0])
            batch_time.update(time.time() - end)
            end = time.time()

            if i != 0 and i % args.print_freq == 0:
                logger.info("EVAL, [{step} / {total_step}]  Loss: {losses.avg:.4e}, "
                            "Batch: {batch_time.sum:.1f}s, "
                            "Data: {data_time.sum:.1f}s, "
                            "Acc@1: {top1.avg:.3f}%, "
                            "Acc@5: {top5.avg:.3f}%  "
                            .format(step=i, total_step=total_step, losses=losses,
                                   batch_time=batch_time, data_time=data_time,
                                   top1=top1, top5=top5))

    logger.info("EVAL, Acc@1: {top1.avg:.3f}%, Acc@5: {top5.avg:.3f}%".format(top1=top1, top5=top5))

    return top1.avg, top5.avg

if __name__ == "__main__":
    main()

