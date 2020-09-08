import sys
import torch
from pathlib import Path
from network import wristNetwork, wristTrocarNetwork
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_model, trocarTester

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
out_joints = [3,4,5]
num_joints = len(out_joints)
in_joints = [3,4,5]
window = 20
skip = 5
joints = [3,4,5]
root = Path('checkpoints' ) 

#############################################
## Load free space model
#############################################

epoch_to_use = 1000
folder = "free_space_wrist_window"+str(window) + "_" + str(skip)
fs_networks = []
for j in range(num_joints):
    fs_networks.append(wristNetwork(window, num_joints))
    fs_networks[j] = load_model(root, folder, epoch_to_use, fs_networks[j], j, device)

#############################################
## Load trocar model
#############################################

data = "trocar"
folder = data + "_wrist_2_part_"+str(window) + '_' + str(skip)

batch_size = 1000000
epoch_to_use = int(sys.argv[1])

networks = []
for j in range(num_joints):
    networks.append(wristTrocarNetwork(window, num_joints))
    networks[j] = load_model(root, folder, epoch_to_use, networks[j], j, device)

model = trocarTester(data, networks, window, skip, out_joints, in_joints, batch_size, device, fs_networks)
uncorrected_loss, corrected_loss = model.test()

print('Uncorrected loss: t4=%f, t5=%f, t6=%f, mean=%f' % (uncorrected_loss[0], uncorrected_loss[1], uncorrected_loss[2], (torch.mean(uncorrected_loss))))
print('Corrected loss: t4=%f, t5=%f, t6=%f, mean=%f' % (corrected_loss[0], corrected_loss[1], corrected_loss[2], (torch.mean(corrected_loss))))
