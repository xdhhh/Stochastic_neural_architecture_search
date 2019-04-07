import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):
  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.COSTS = []
    self.height = 32
    self.width = 32
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'sep' in primitive:
        FLOP = 0;MAC = 0
        FLOP1 = (self.height * self.width * op.op[1].kernel_size[0]**2 *op.op[1].in_channels * op.op[1].out_channels )
        FLOP1 = FLOP / op.op[1].groups
        FLOP2 = (self.height * self.width * op.op[1].kernel_size[0]**2)
        FLOP = (FLOP1 + FLOP2)*2
        MAC1 = self.height*self.width*(op.op[1].in_channels + op.op[1].out_channels)
        MAC1 = MAC1 + (op.op[1].in_channels * op.op[1].out_channels)
        MAC2 = MAC1 + (op.op[1].in_channels * op.op[1].out_channels)/ op.op[1].groups
        MAC = (MAC1 + MAC2)*2

      if 'dil' in primitive:
        FLOP = (self.width*self.height*op.op[1].in_channels*op.op[1].out_channels*op.op[1].kernel_size[0]**2) / op.op[1].groups
        FLOP +=(self.width*self.height*op.op[2].in_channels*op.op[2].out_channels)

        MAC1 = (self.width * self.height)*(op.op[1].in_channels + op.op[1].in_channels)
        MAC1 += (op.op[1].kernel_size[0]**2 *op.op[1].in_channels* op.op[1].out_channels) / op.op[1].groups
        MAC2 = (self.width * self.height)*(op.op[2].in_channels + op.op[2].in_channels)
        MAC2 += (op.op[2].kernel_size[0]**2 *op.op[2].in_channels* op.op[2].out_channels) / op.op[2].groups

        MAC = MAC1 + MAC2
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      if 'skip' in primitive:
        FLOP = 0
        MAC = 2*C * self.height * self.width
      if 'none' in primitive:
        FLOP = 0
        MAC = 0
      self._ops.append(op)
      self.COSTS.append(FLOP + MAC)

  def forward(self, x, Z):
    self.height = x.shape[0]
    self.width = x.shape[1]
    return sum(z * op(x) for z, op in zip(Z, self._ops)) , sum(a for a in self.COSTS)





class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op  = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, Z):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    costs = []
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, Z[offset+j])[0] for j, h in enumerate(states))
      cost = sum(self._ops[offset+j](h, Z[offset+j])[1] for j, h in enumerate(states))
      offset += len(states)
      states.append(s)
      costs.append(cost)

    return torch.cat(states[-self._multiplier:], dim=1),torch.sum(torch.tensor(costs[-self._multiplier:]))



class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network,self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
        nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
        nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def forward(self, input , temperature):
        s0 = s1 = self.stem(input)
        costs = 0
        score = 0
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                Z = self.ArchitectDist(self.alphas_reduce , temperature) # shape = [14,8]
            else:
                Z = self.ArchitectDist(self.alphas_normal , temperature)
            s2 ,cost = cell(s0,s1,Z)
            s0, s1 = s1 ,s2
            costs += cost
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits ,costs


    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters


    def ArchitectDist(self, alpha, temperature):
        weight = nn.functional.gumbel_softmax(alpha, temperature)
        return weight
