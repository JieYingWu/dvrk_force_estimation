import sys
import torch
from pathlib import Path
from dataset import indirectDataset
from network import torqueNetwork, trocarNetwork
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import nrmse_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
JOINTS = 6
window = 10
skip = 10
root = Path('checkpoints' ) 

#############################################
## Load free space model
#############################################

epoch_to_use = 1000
free_space_networks = []
for j in range(JOINTS):
    free_space_networks.append(torqueNetwork(window))
    free_space_networks[j].to(device)

model_root = root / "models_indirect" / ("free_space_window"+str(window) + "_" + str(skip))
for j in range(JOINTS):
    model_path = model_root / 'model_joint_{}_{}.pt'.format(j, epoch_to_use)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch'] + 1
        free_space_networks[j].load_state_dict(state['model'])
        free_space_networks[j].eval()
        for param in free_space_networks[j].parameters():
            param.requires_grad = False
        print('Restored model, epoch {}, joint {}'.format(epoch-1, j))
    else:
        print('Failed to restore model')
        exit()

#############################################
## Load trocar model
#############################################

data = "trocar"
test_path = '../data/csv/test/' + data + '/no_contact'
folder = data + "_2_part_"+str(window) + '_' + str(skip)

batch_size = 1000000
epoch_to_use = 1000

networks = []
for j in range(JOINTS):
    networks.append(trocarNetwork(window))
    networks[j].to(device).eval()

test_dataset = indirectDataset(test_path, window, skip)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=False)

model_root = root / "models_indirect" / folder
for j in range(JOINTS):
    model_path = model_root / 'model_joint_{}_{}.pt'.format(j, epoch_to_use)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch'] + 1
        networks[j].load_state_dict(state['model'])
        print('Restored model, epoch {}, joint {}'.format(epoch-1, j))
    else:
        print('Failed to restore model')
        exit()

test_loss = torch.zeros(JOINTS)
for i, (posvel, torque, jacobian) in enumerate(test_loader):
    posvel = posvel.to(device)
    torque = torque.to(device)

    free_space_torque = torch.zeros(posvel.size()[0], 6).to(device)
    for j in range(JOINTS):
        free_space_torque[:,j] = free_space_networks[j](posvel).squeeze()
        
    for j in range(JOINTS):
#        pred = networks[j](posvel)
        loss = nrmse_loss(free_space_torque[:,j], torque[:,j])
        test_loss[j] += loss.item()
        
print('Test loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4], test_loss[5], (torch.mean(test_loss)/len(test_loader))))
