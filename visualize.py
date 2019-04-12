import numpy as np
import torch
from genotypes import *
from visualize_ftn import *
from option.default_option import TrainOptions

epoch = int(120)

temperature = 2.5
alpha_normal = np.load('search_results_3/alpha_normal_' + str(epoch) + '.npy')
alpha_reduce = np.load('search_results_3/alpha_reduce_' + str(epoch) + '.npy')


m_normal = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
    torch.tensor([temperature]), torch.tensor(alpha_normal))
m_reduce = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
    torch.tensor([temperature]) , torch.tensor(alpha_reduce))

alpha_normal = m_normal.sample().cpu().numpy()
alpha_reduce = m_reduce.sample().cpu().numpy()
ex = genotype(alpha_normal,alpha_reduce)
plot(ex.normal,'normal.pdf')
plot(ex.reduce,'reduce.pdf')
