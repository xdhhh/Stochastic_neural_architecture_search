import numpy as np
import utils
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
from torch.autograd import Variable
from model_search_cons import Network
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import argparse

def train(train_queue,valid_queue, model, criterion, optimizer_arch, optimizer_model,lr_arch, lr_model, temperature):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        n = input.size(0) # batch size

        input = Variable(input , requires_grad = True).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)

        optimizer_arch.zero_grad()
        optimizer_model.zero_grad()
        logit,cost= model(input, temperature)## model inputs
        value_loss = criterion(logit , target)
        total_loss = value_loss + Variable(cost*(1e-9)).cuda()
        total_loss.backward()
        nn.utils.clip_grad_norm(model.parameters(),5)
        optimizer_arch.step()
        optimizer_model.step()

        prec1, prec5 = utils.accuracy(logit, target, topk=(1, 5))
        objs.update(value_loss.data, n)
        top1.update(prec1.data , n)
        top5.update(prec5.data , n)
    return top1.avg, top5.avg, objs.avg


def infer(valid_queue, model, criterion, temperature):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(async=True)

        logits ,cost = model(input , temperature)
        loss = criterion(logits, target)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data , n)
        top1.update(prec1.data , n)
        top5.update(prec5.data , n)
    return top1.avg, top5.avg ,objs.avg

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--device_number', type=str, default='0', help='device number')
parser.add_argument('--tensorboard_log', type=str, default='log', help='tensorboard log')
parser.add_argument('--search_results_dir', type=str, default='search_results', help='search results dir')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--tem_decay', type=float, default=0.97, help='temperature decay ratio')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--epochs', type=int, default=200, help='epochs')
parser.add_argument('--init_channels', type=int, default=16, help='initial channels')
parser.add_argument('--layers', type=int, default=8, help='layers')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--initial_temp', type=float, default=2.5, help='innitial softmax temperature')
parser.add_argument('--decay_ratio', type=float, default=0.00003, help='annealation rate of softmax temperature')
parser.add_argument('--train_portion', type=float, default=0.5, help='train samples portion')
args = parser.parse_args()
lr_scheduler = {
  'T_max' : 400,
  'alpha' : 1e-4,
  'warmup_step' : 100,
  't_mul' : 1.5,
  'lr_mul' : 0.98,
}
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
device = torch.device('cuda')
CIFAR_CLASSES = 10
criterion = nn.CrossEntropyLoss().cuda()
model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
model.cuda()
optimizer_model = torch.optim.SGD(model.parameters(),lr= 0.025,momentum = 0.9, weight_decay=3e-4)
optimizer_arch = torch.optim.Adam(model.arch_parameters(),lr = 3e-4, betas=(0.5, 0.999), weight_decay = 1e-3)
w_sche = utils.CosineDecayLR(optimizer_model, **lr_scheduler)
train_transform, valid_transform = utils._data_transforms_cifar10(args)
train_data = dset.CIFAR10(root='/home/xiaoda/data', train=True, download=False, transform=train_transform)
valid_data = dset.CIFAR10(root='/home/xiaoda/data', train=False, download=False, transform=train_transform)
writer = SummaryWriter(args.tensorboard_log)
if not os.path.exists(args.search_results_dir):
    os.mkdir(args.search_results_dir)
train_queue = torch.utils.data.DataLoader(
  train_data, batch_size=args.batch_size, shuffle = True)

valid_queue = torch.utils.data.DataLoader(
  valid_data, batch_size=args.batch_size, shuffle = False)

temperature = args.initial_temp

for epoch in range(args.epochs):
    print("Start to train for epoch %d" % (epoch))
    train_acc_top1, train_acc_top5 , train_valoss = train(train_queue, valid_queue, model,criterion, optimizer_arch,optimizer_model, 3e-4,0.025, temperature)
    w_sche.step()
    print("Start to validate for epoch %d" % (epoch))
    valid_acc_top1,valid_acc_top5, valid_valoss = infer(valid_queue, model, criterion, temperature)
    temperature = utils.decay_temperature(temperature, args.tem_decay)
    writer.add_scalar('Train/train_acc_top1', train_acc_top1, epoch)
    writer.add_scalar('Train/train_acc_top5', train_acc_top5, epoch)
    writer.add_scalar('Train/train_valoss', train_valoss, epoch)
    writer.add_scalar('Val/valid_acc_top1', valid_acc_top1, epoch)
    writer.add_scalar('Val/valid_acc_top5', valid_acc_top5, epoch)
    writer.add_scalar('Val/valid_valoss', valid_valoss, epoch)
    if epoch % 10 ==0:
      np.save(args.search_results_dir + "/alpha_normal_" + str(epoch) + ".npy"  , model.alphas_normal.detach().cpu().numpy())
      np.save(args.search_results_dir + "/alpha_reduce_" + str(epoch) + ".npy"  , model.alphas_reduce.detach().cpu().numpy())

torch.save(model.state_dict(), args.search_results_dir + '/weights.pt')
