import sys
import torch
import numpy as np
from pathlib import Path
from dataset import indirectDataset
from network import insertionNetwork
from torch.utils.data import DataLoader
from utils import nrmse_loss, load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
window = int(sys.argv[2])
skip = int(sys.argv[3])

data = sys.argv[1]
test_path = '../data/csv/test/' + data + '/no_contact/'
root = Path('checkpoints' )
folder = data + "_insertion_window"+str(window)+'_'+str(skip)

batch_size = 1000000
epoch_to_use = int(sys.argv[4])
loss_fn = torch.nn.MSELoss()

network = insertionNetwork(window).to(device)
network = load_model(root, folder, epoch_to_use, network, 0, device)

test_dataset = indirectDataset(test_path, window, skip)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size)
test_loss = 0
    
for i, (position, velocity,  torque, jacobian) in enumerate(test_loader):
    position = position.to(device)
    velocity = velocity.to(device)
    posvel = torch.cat((position, velocity), axis=1)
    torque = torque.to(device)

    pred = network(posvel).detach()
    loss = nrmse_loss(pred.squeeze(), torque[:,2]).detach()
    test_loss += loss.item()
        
print('Test loss: f3=%f' % (test_loss))
