import sys
import torch
import numpy as np
from pathlib import Path
from dataset import indirectDataset
from network import armNetwork, insertionNetwork, wristNetwork, armTrocarNetwork, insertionTrocarNetwork, wristTrocarNetwork
from torch.utils.data import DataLoader
from utils import nrmse_loss, load_model, trocarTester

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = 'trocar'
root = Path('checkpoints' )
batch_size = 1000000
loss_fn = nrmse_loss #torch.nn.MSELoss()
uncorrected_loss = torch.zeros(6)
corrected_loss = torch.zeros(6)
epoch = 1000
trocar_epoch = (sys.argv[1])

#############################################
## Load free space and trocar arm model
#############################################

out_joints = [0,1]
num_joints = len(out_joints)
in_joints = [0,1,2,3,4,5]
window = 10
skip = 2

fs_folder = "free_space_arm_window"+str(window)+'_'+str(skip)
trocar_folder = "trocar_arm_2_part_"+str(window) + '_' + str(skip)

fs_networks= []
networks = []
for j in range(num_joints):
    fs_networks.append(armNetwork(window))
    networks.append(armTrocarNetwork(window))

    fs_networks[j] = load_model(root, fs_folder, epoch, fs_networks[j], j, device)
    networks[j] = load_model(root, trocar_folder, trocar_epoch, networks[j], j, device)

#############################################
## Load dataset and run arm models
#############################################
        
model = trocarTester(data, networks, window, skip, out_joints, in_joints, batch_size, device, fs_networks)
uncorrected_loss[0:2], corrected_loss[0:2] = model.test()

#############################################
## Load free space and trocar insertion model
#############################################

out_joints = [2]
num_joints = len(out_joints)
in_joints = [0,1,2,3,4,5]
window = 50
skip = 2

fs_folder = "free_space_insertion_window"+str(window)+'_'+str(skip)
trocar_folder = "trocar_insertion_2_part_"+str(window) + '_' + str(skip)

fs_networks = insertionNetwork(window)
fs_networks = [load_model(root, fs_folder, epoch, fs_networks, 0, device)]

#############################################
## Load dataset and run insertion models
#############################################

networks = []
for j in range(num_joints):
    networks.append(insertionTrocarNetwork(window))
    networks[j] = load_model(root, trocar_folder, trocar_epoch, networks[j], j, device)

model = trocarTester(data, networks, window, skip, out_joints, in_joints, batch_size, device, fs_networks)
uncorrected_loss[2], corrected_loss[2] = model.test()

#############################################
## Load free space and trocar wrist model
#############################################

window = 2
skip = 1
out_joints = [3,4,5]
num_joints = len(out_joints)
in_joints = [3,4,5]

fs_folder = "free_space_wrist_window"+str(window)+'_'+str(skip)
trocar_folder = "trocar_wrist_2_part_"+str(window) + '_' + str(skip)

fs_networks = []
networks = []

for j in range(num_joints):
    fs_networks.append(wristNetwork(window, len(in_joints)))
    networks.append(wristTrocarNetwork(window, len(in_joints)))

    fs_networks[j] = load_model(root, fs_folder, epoch, fs_networks[j], j, device)
    networks[j] = load_model(root, trocar_folder, trocar_epoch, networks[j], j, device)

#############################################
## Load dataset and run wrist models
#############################################

networks = []
for j in range(num_joints):
    networks.append(wristTrocarNetwork(window, num_joints))
    networks[j] = load_model(root, trocar_folder, trocar_epoch, networks[j], j, device)

model = trocarTester(data, networks, window, skip, out_joints, in_joints, batch_size, device, fs_networks)
uncorrected_loss[3:6], corrected_loss[3:6] = model.test()

#############################################
## Save results and print out
#############################################

print('Uncorrected loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (uncorrected_loss[0], uncorrected_loss[1], uncorrected_loss[2], uncorrected_loss[3], uncorrected_loss[4], uncorrected_loss[5], torch.mean(uncorrected_loss)))

print('Corrected loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (corrected_loss[0], corrected_loss[1], corrected_loss[2], corrected_loss[3], corrected_loss[4], corrected_loss[5], torch.mean(corrected_loss)))
