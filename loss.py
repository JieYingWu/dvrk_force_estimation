import torch
import numpy as np
from torch import nn as nn

# The fact that they're over all the samples makes it hard to learn from these losses
# Better used for testing
class NmrseLoss(nn.Module):

    def __init__(self):
        super(NmrseLoss, self).__init__()

    def forward(self, y_hat, y):
        denominator = y.max()-y.min()
        summation = torch.sum((y_hat-y)**2)
        nrmse = torch.sqrt((summation/y.size()[0]))/denominator
        return nrmse
 
class RelELoss(nn.Module):

    def __init__(self):
        super(RelELoss, self).__init__()

    def forward(self, y_hat, y):
        nominator = torch.sum((y_hat-y)**2)
        demonimator = torch.sum(y**2)
        error = torch.sqrt(nominator/denominator)
        return error
 
