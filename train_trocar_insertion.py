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
in_joints = [0,1,2,3,4,5]
batch_size = 4096
root = Path('checkpoints') 

#############################################
## Load free space model
#############################################

epoch_to_use = 1000
folder = "free_space_insertion_window"+str(window) + "_" + str(skip)
fs_network = insertionNetwork(window)
fs_network = load_model(root, folder, epoch_to_use, fs_network, device)

#############################################
## Set up trocar model to train
#############################################

data = "trocar"
train_path = '../data/csv/train/' + data + '/'
val_path = '../data/csv/val/' + data + '/'
folder = data + "_insertion_2_part_"+str(window) + '_' + str(skip)

lr = 1e-2
epochs = 1000
validate_each = 5
use_previous_model = False
epoch_to_use = 250

network = insertionTrocarNetwork(window)
model = trocarLearner(data, folder, network, window, skip, out_joints, in_joints, batch_size, lr, device, fs_network)
    
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
