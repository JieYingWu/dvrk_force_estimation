import sys
import torch
import numpy as np
from pathlib import Path
from dataset import indirectDataset
from network import armNetwork, insertionNetwork, wristNetwork, armTrocarNetwork, insertionTrocarNetwork, wristTrocarNetwork
from torch.utils.data import DataLoader
from utils import nrmse_loss, load_model, trocarTester

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1000000
uncorrected_loss = torch.zeros(6)
corrected_loss = torch.zeros(6)
epoch = 1000
trocar_epoch = (sys.argv[1])

#####################################################
## Load free space and trocar arm model and run
#####################################################

out_joints = [0,1]
in_joints = [0,1,2,3,4,5]
window = 10
skip = 2

fs_folder = "free_space_arm_window"+str(window)+'_'+str(skip)
trocar_folder = "trocar_arm_2_part_"+str(window) + '_' + str(skip)

fs_network = armNetwork(window)
fs_network = load_model(fs_folder, epoch, fs_network, device)
network = armTrocarNetwork(window) 
        
model = trocarTester("trocar", trocar_folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network)
model.load_prev(trocar_epoch)
uncorrected_loss[0:2], corrected_loss[0:2], torque, fs_pred, pred = model.test()

#####################################################
## Load free space and trocar insertion model and run
#####################################################

out_joints = [2]
in_joints = [0,1,2,3,4,5]
window = 50
skip = 2

fs_folder = "free_space_insertion_window"+str(window)+'_'+str(skip)
trocar_folder = "trocar_insertion_2_part_"+str(window) + '_' + str(skip)

fs_network = insertionNetwork(window)
fs_network = load_model(fs_folder, epoch, fs_network, device)
network = insertionTrocarNetwork(window)
model = trocarTester("trocar", trocar_folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network)
model.load_prev(trocar_epoch)
uncorrected_loss[2], corrected_loss[2], torque, fs_pred, pred = model.test()

#############################################
## Load free space and trocar wrist model
#############################################

window = 5
skip = 1
out_joints = [3,4,5]
in_joints = [3,4,5]

fs_folder = "free_space_wrist_window"+str(window)+'_'+str(skip)
trocar_folder = "trocar_wrist_2_part_"+str(window) + '_' + str(skip)

fs_network = wristNetwork(window, len(in_joints))
fs_network = load_model(fs_folder, epoch, fs_network, device)

network = wristTrocarNetwork(window, len(in_joints))
model = trocarTester("trocar", trocar_folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network)
model.load_prev(trocar_epoch)
uncorrected_loss[3:6], corrected_loss[3:6], torque, fs_pred, pred = model.test()

#############################################
## Save results and print out
#############################################

print('Uncorrected loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (uncorrected_loss[0], uncorrected_loss[1], uncorrected_loss[2], uncorrected_loss[3], uncorrected_loss[4], uncorrected_loss[5], torch.mean(uncorrected_loss)))

print('Corrected loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (corrected_loss[0], corrected_loss[1], corrected_loss[2], corrected_loss[3], corrected_loss[4], corrected_loss[5], torch.mean(corrected_loss)))
