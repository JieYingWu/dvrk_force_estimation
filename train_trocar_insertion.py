import sys
import tqdm
import torch
from pathlib import Path
from dataset import indirectDataset
from network import insertionNetwork, insertionTrocarNetwork
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
window = 50
skip = 2
out_joints = [2]
num_joints = len(out_joints)
in_joints = [0,1,2,3,4,5]
root = Path('checkpoints') 

#############################################
## Load free space model
#############################################

epoch_to_use = 1000
free_space_networks = []
for j in range(num_joints):
    folder = "free_space_insertion_window"+str(window) + "_" + str(skip)
    free_space_networks.append(insertionNetwork(window))
    free_space_networks[j] = load_model(root, folder, epoch_to_use, free_space_networks[j], j, device)

#############################################
## Set up trocar model to train
#############################################

data = "trocar"
train_path = '../data/csv/train/' + data + '/'
val_path = '../data/csv/val/' + data + '/'
folder = data + "_insertion_2_part_"+str(window) + '_' + str(skip)

lr = 1e-2
batch_size = 4096
epochs = 1000
validate_each = 5
use_previous_model = False
epoch_to_use = 250

networks = []
for j in range(num_joints):
    networks.append(insertionTrocarNetwork(window).to(device))
                          
model = trocarLearner(data, folder, networks, window, skip, out_joints, in_joints, batch_size, lr, device, free_space_networks)
    
if use_previous_model:
    model.load_prev(epoch_to_use)
    epoch = epoch_to_use + 1
else:
    epoch = 1
    

print('Training for ' + str(epochs))
for e in range(epoch, epochs + 1):

    model.train_step(e)
    
    if e % validate_each == 0:
        model.val_step(e)
