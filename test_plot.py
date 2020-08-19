import sys
import torch
import numpy as np
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
batch_size = 1000000
epoch_to_use = 1000

data = sys.argv[1]
test_path = '../data/csv/test/' + data + '/no_contact'
test_dataset = indirectDataset(test_path, window, skip)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=False)

#############################################
## Load free space model
#############################################

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

trocar_networks = []
for j in range(JOINTS):
    trocar_networks.append(torqueNetwork(window))
    trocar_networks[j].to(device).eval()

model_root = root / "models_indirect" / ("trocar_window"+str(window) + "_" + str(skip))

for j in range(JOINTS):
    model_path = model_root / 'model_joint_{}_{}.pt'.format(j, epoch_to_use)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch'] + 1
        trocar_networks[j].load_state_dict(state['model'])
        print('Restored model, epoch {}, joint {}'.format(epoch-1, j))
    else:
        print('Failed to restore model')
        exit()

#############################################
## Run both models
#############################################
        
free_space_loss = torch.zeros(JOINTS)
trocar_loss = torch.zeros(JOINTS)
for i, (posvel, torque, jacobian) in enumerate(test_loader):
    posvel = posvel.to(device)
    torque = torque.to(device)

    free_space_pred = torch.zeros(torque.size()).to(device)
    trocar_pred = torch.zeros(torque.size()).to(device)
    for j in range(JOINTS):
        free_space_pred[:,j] = free_space_networks[j](posvel).squeeze()
        loss = nrmse_loss(free_space_pred[:,j].squeeze(), torque[:,j])
        free_space_loss[j] += loss.item()
        
        trocar_pred[:,j] = trocar_networks[j](posvel).squeeze()
        loss = nrmse_loss(trocar_pred[:,j].squeeze(), torque[:,j])
        trocar_loss[j] += loss.item()



all_torque = torch.cat((torque, free_space_pred.detach(), trocar_pred.detach()), axis=1).cpu().numpy()
np.savetxt('pred.csv', all_torque, delimiter=',')
                
print('Free space loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (free_space_loss[0], free_space_loss[1], free_space_loss[2], free_space_loss[3], free_space_loss[4], free_space_loss[5], (torch.mean(free_space_loss)/len(test_loader))))
print('Trocar loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (trocar_loss[0], trocar_loss[1], trocar_loss[2], trocar_loss[3], trocar_loss[4], trocar_loss[5], (torch.mean(trocar_loss)/len(test_loader))))
