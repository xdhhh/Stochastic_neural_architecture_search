import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
import math

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def _data_transforms_cifar10(opt):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if opt.cutout:
    train_transform.transforms.append(Cutout(opt.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x

def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res

class CosineDecayLR(_LRScheduler):
  def __init__(self, optimizer, T_max, alpha=1e-4,
               t_mul=2, lr_mul=0.9,
               last_epoch=-1,
               warmup_step=300):
    self.T_max = T_max
    self.alpha = alpha
    self.t_mul = t_mul
    self.lr_mul = lr_mul
    self.warmup_step = warmup_step
    self.last_restart_step = 0
    self.flag = True
    super(CosineDecayLR, self).__init__(optimizer, last_epoch)

    self.min_lrs = [b_lr * alpha for b_lr in self.base_lrs]
    self.rise_lrs = [1.0 * (b - m) / self.warmup_step
                     for (b, m) in zip(self.base_lrs, self.min_lrs)]

  def get_lr(self):
    T_cur = self.last_epoch - self.last_restart_step
    assert T_cur >= 0
    if T_cur <= self.warmup_step and (not self.flag):
      base_lrs = [min_lr + rise_lr * T_cur
              for (base_lr, min_lr, rise_lr) in
                zip(self.base_lrs, self.min_lrs, self.rise_lrs)]
      if T_cur == self.warmup_step:
        self.last_restart_step = self.last_epoch
        self.flag = True
    else:
      base_lrs = [self.alpha + (base_lr - self.alpha) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs]
    if T_cur == self.T_max:
      self.last_restart_step = self.last_epoch
      self.min_lrs = [b_lr * self.alpha for b_lr in self.base_lrs]
      self.base_lrs = [b_lr * self.lr_mul for b_lr in self.base_lrs]
      self.rise_lrs = [1.0 * (b - m) / self.warmup_step
                     for (b, m) in zip(self.base_lrs, self.min_lrs)]
      self.T_max = int(self.T_max * self.t_mul)
      self.flag = False

    return base_lrs
