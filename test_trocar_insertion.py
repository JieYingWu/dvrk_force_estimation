import sys
import torch
from pathlib import Path
from dataset import indirectDataset
from network import insertionNetwork, insertionTrocarNetwork
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import nrmse_loss, load_model, trocarTester

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
window = 50
skip = 2
out_joints = [2]
num_joints = len(out_joints)
in_joints = [0,1,2,3,4,5]
root = Path('checkpoints' ) 

#############################################
## Load free space model
#############################################

epoch_to_use = 1000
folder = "free_space_insertion_window"+str(window) + "_" + str(skip)
fs_networks = insertionNetwork(window)
fs_networks = [load_model(root, folder, epoch_to_use, fs_networks, 0, device)]

#############################################
## Load trocar model
#############################################

data = "trocar"
test_path = '../data/csv/test/' + data + '/no_contact'
folder = data + "_insertion_2_part_"+str(window) + '_' + str(skip)

batch_size = 1000000
epoch_to_use = int(sys.argv[1])

networks = []
for j in range(num_joints):
    networks.append(insertionTrocarNetwork(window))
    networks[j] = load_model(root, folder, epoch_to_use, networks[j], j, device)

model = trocarTester(data, networks, window, skip, out_joints, in_joints, batch_size, device, fs_networks)
uncorrected_loss, corrected_loss = model.test()

print('Uncorrected loss: f3=%f, mean=%f' % (uncorrected_loss[0], (torch.mean(uncorrected_loss))))
print('Corrected loss: f3=%f, mean=%f' % (corrected_loss[0], (torch.mean(corrected_loss))))
