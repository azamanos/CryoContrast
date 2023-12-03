from __future__ import print_function

import os
import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

class SupervisedDataset(Dataset):
    def __init__(self, X, y, transform):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.transform(self.X[index]), self.y[index]
    

def load_xy(dir):
    positives_dir = os.path.join(dir, "positives")
    negatives_dir = os.path.join(dir, "negatives")

    x_pos = np.load(os.path.join(positives_dir, "X.npy"))
    y_pos  = np.ones(x_pos.shape[0])

    x_neg = np.load(os.path.join(negatives_dir, "X.npy"))
    y_neg = np.zeros(x_neg.shape[0])

    x = np.concatenate((x_pos.astype(np.float32), x_neg.astype(np.float32))) + 128
    y = np.concatenate((y_pos, y_neg))

    return x[:100], y[:100]


def load_dataset(dir, transform):

    train_dir = os.path.join(dir, "train")
    val_dir = os.path.join(dir, "validation")
    test_dir = os.path.join(dir, "test")

    x, y = load_xy(train_dir)
    train_dataset = SupervisedDataset(x, y, transform=transform)
    """
    x, y = load_xy(val_dir)
    val_dataset = SupervisedDataset(x, y)
    x, y = load_xy(test_dir)
    test_dataset = SupervisedDataset(x, y)
    """
    return train_dataset, None, None #val_dataset, test_dataset