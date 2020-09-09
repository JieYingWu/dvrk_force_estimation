import sys
import torch
from pathlib import Path
from network import wristNetwork, wristTrocarNetwork
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_model, trocarTester
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
out_joints = [3,4,5]
num_joints = len(out_joints)
in_joints = [3,4,5]
window = 2
skip = 1
joints = [3,4,5]
root = Path('checkpoints' ) 

#############################################
## Load free space model
#############################################

epoch_to_use = 1000
folder = "free_space_wrist_window"+str(window) + "_" + str(skip)
fs_network = wristNetwork(window, len(in_joints))
fs_network = load_model(root, folder, epoch_to_use, fs_network, device)

#############################################
## Load trocar model
#############################################

data = "trocar"
folder = data + "_wrist_2_part_"+str(window) + '_' + str(skip)

batch_size = 1000000
epoch_to_use = int(sys.argv[1])

network = wristTrocarNetwork(window, len(in_joints))
model = trocarTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network)
model.load_prev(epoch_to_use)
uncorrected_loss, corrected_loss, torque, fs_pred, pred = model.test(True)

all_torque = torch.cat((torque, fs_pred, pred), axis=1).numpy()
np.savetxt('generated/wrist_torcar_pred.csv', all_torque, delimiter=',')

print('Uncorrected loss: t4=%f, t5=%f, t6=%f, mean=%f' % (uncorrected_loss[0], uncorrected_loss[1], uncorrected_loss[2], (torch.mean(uncorrected_loss))))
print('Corrected loss: t4=%f, t5=%f, t6=%f, mean=%f' % (corrected_loss[0], corrected_loss[1], corrected_loss[2], (torch.mean(corrected_loss))))
