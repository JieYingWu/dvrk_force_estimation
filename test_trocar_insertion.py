import sys
import torch
from pathlib import Path
from dataset import indirectDataset
from network import insertionNetwork, insertionTrocarNetwork
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import nrmse_loss, load_model, trocarTester
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
window = 50
skip = 2
out_joints = [2]
in_joints = [0,1,2,3,4,5]
root = Path('checkpoints' ) 

#############################################
## Load free space model
#############################################

epoch_to_use = 1000
folder = "free_space_insertion_window"+str(window) + "_" + str(skip)
fs_network = insertionNetwork(window)
fs_network = load_model(root, folder, epoch_to_use, fs_network, device)

#############################################
## Load trocar model
#############################################

data = "trocar"
test_path = '../data/csv/test/' + data + '/no_contact'
folder = data + "_insertion_2_part_"+str(window) + '_' + str(skip)

batch_size = 1000000
epoch_to_use = int(sys.argv[1])

network = insertionTrocarNetwork(window)
model = trocarTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network)
model.load_prev(epoch_to_use)
uncorrected_loss, corrected_loss, torque, fs_pred, pred = model.test(True)

all_torque = torch.cat((torque, fs_pred, pred), axis=1).numpy()
np.savetxt('generated/insertion_torcar_pred.csv', all_torque, delimiter=',')

print('Uncorrected loss: f3=%f, mean=%f' % (uncorrected_loss[0], (torch.mean(uncorrected_loss))))
print('Corrected loss: f3=%f, mean=%f' % (corrected_loss[0], (torch.mean(corrected_loss))))
