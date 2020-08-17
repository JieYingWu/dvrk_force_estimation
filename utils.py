import torch
import torch.nn as nn

def nrmse_loss(y_hat, y):
    print(y.max(), y.min())
    print(y_hat.max(), y_hat.min())
    denominator = y.max()-y.min()
    summation = torch.sum((y_hat-y)**2)
    nrmse = torch.sqrt((summation/y.size()[0]))/denominator
    return nrmse * 100

def relELoss(y_hat, y):
    nominator = torch.sum((y_hat-y)**2)
    demonimator = torch.sum(y**2)
    error = torch.sqrt(nominator/denominator)
    return error * 100

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

