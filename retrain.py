import numpy as np
import utils
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
from torch.autograd import Variable
from model import NetworkCIFAR as Network
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import argparse
from visualize_ftn import genotype

def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda(async=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
          loss_aux = criterion(logits_aux, target)
          loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()

        prec1 = utils.accuracy(logits, target)
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
    return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()
    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(async=True)
        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1 = utils.accuracy(logits, target)
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
    return top1.avg, objs.avg


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--device_number', type=str, default='0', help='device number')
parser.add_argument('--tensorboard_log', type=str, default='log', help='tensorboard log')
parser.add_argument('--search_results_dir', type=str, default='search_results', help='search results dir')
parser.add_argument('--results_dir', type=str, default='results', help='results dir')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight_decay')
parser.add_argument('--epochs', type=int, default=600, help='epochs')
parser.add_argument('--init_channels', type=int, default=36, help='initial channels')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--layers', type=int, default=20, help='layers')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
device = torch.device('cuda')
CIFAR_CLASSES = 10
criterion = nn.CrossEntropyLoss().cuda()
epoch_ = int(120)
alpha_normal = np.load(args.search_results_dir + '/alpha_normal_' + str(epoch_) + '.npy')
alpha_reduce = np.load(args.search_results_dir + '/alpha_reduce_' + str(epoch_) + '.npy')
ex = genotype(alpha_normal,alpha_reduce)
model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, ex)
model.cuda()
optimizer = torch.optim.SGD(model.parameters(),lr= args.learning_rate,momentum = args.momentum, weight_decay=args.weight_decay)
train_transform, valid_transform = utils._data_transforms_cifar10(args)
train_data = dset.CIFAR10(root='/home/xiaoda/data', train=True, download=False, transform=train_transform)
valid_data = dset.CIFAR10(root='/home/xiaoda/data', train=False, download=False, transform=train_transform)
writer = SummaryWriter(args.tensorboard_log)
if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
train_queue = torch.utils.data.DataLoader(
  train_data, batch_size=args.batch_size, shuffle = True)

valid_queue = torch.utils.data.DataLoader(
  valid_data, batch_size=args.batch_size, shuffle = False)
for epoch in range(args.epochs):
    scheduler.step()
    print("Start to train for epoch %d" % (epoch))
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    print("Start to validate for epoch %d" % (epoch))
    valid_acc, valid_obj = infer(valid_queue, model, criterion)

    writer.add_scalar('Train/train_acc', train_acc, epoch)
    writer.add_scalar('Train/train_loss', train_obj, epoch)
    writer.add_scalar('Val/valid_acc', valid_acc, epoch)
    writer.add_scalar('Val/valid_loss', valid_obj, epoch)

torch.save(model.state_dict(), args.results_dir + '/weights.pt')
