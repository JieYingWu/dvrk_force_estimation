import os
import sys
import torch
import numpy as np
from os.path import join
from dataset import indirectDataset
from network import *
from torch.utils.data import DataLoader
from utils import nrmse_loss, load_model, trocarTester

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1000000
uncorrected_loss = torch.zeros(7)
corrected_loss = torch.zeros(7)
epoch = 1000
trocar_epoch = (sys.argv[1])
test = 'test'
contact = 'no_contact'
exp = 'exp1'
path = join('..', 'data', 'csv', test, 'trocar', contact, exp)

#####################################################
## Load free space and trocar arm model and run
#####################################################

out_joints = [0,1]
in_joints = [0,1,2,3,4,5]
window = 10
skip = 2

fs_folder = "free_space_arm_window"+str(window)+'_'+str(skip)
trocar_folder = "trocar_arm_2_part_"+str(window) + '_' + str(skip)

fs_network = armNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
fs_network = load_model(fs_folder, epoch, fs_network, device)
network = trocarNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
        
model = trocarTester("trocar", trocar_folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network, path)
model.load_prev(trocar_epoch)
uncorrected_loss[out_joints], corrected_loss[out_joints], torque, fs_pred, pred, jacobian, time = model.test()
arm_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)
arm_fs_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)

#####################################################
## Load free space and trocar insertion model and run
#####################################################

out_joints = [2]
in_joints = [0,1,2,3,4,5]
window = 50
skip = 2

fs_folder = "free_space_insertion_window"+str(window)+'_'+str(skip)
trocar_folder = "trocar_insertion_2_part_"+str(window) + '_' + str(skip)

fs_network = insertionNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
fs_network = load_model(fs_folder, epoch, fs_network, device)
network = trocarNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
model = trocarTester("trocar", trocar_folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network, path)
model.load_prev(trocar_epoch)
uncorrected_loss[out_joints], corrected_loss[out_joints], torque, fs_pred, pred, jacobian, time = model.test()
insertion_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)
insertion_fs_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)

#####################################################
## Load free space and trocar platform model and run
#####################################################

out_joints = [3]
in_joints = [0,1,2,3,4,5]
window = 5
skip = 1

fs_folder = "free_space_platform_window"+str(window)+'_'+str(skip)
trocar_folder = "trocar_platform_2_part_"+str(window) + '_' + str(skip)

fs_network = wristNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
fs_network = load_model(fs_folder, epoch, fs_network, device)
network = trocarNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
model = trocarTester("trocar", trocar_folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network, path)
model.load_prev(trocar_epoch)
uncorrected_loss[out_joints], corrected_loss[out_joints], torque, fs_pred, pred, jacobian, time = model.test()
platform_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)
platform_fs_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)

#############################################
## Load free space and trocar wrist model
#############################################

window = 5
skip = 1
out_joints = [4,5]
in_joints = [0,1,2,3,4,5]

fs_folder = "free_space_wrist_window"+str(window)+'_'+str(skip)
trocar_folder = "trocar_wrist_2_part_"+str(window) + '_' + str(skip)

fs_network = wristNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
fs_network = load_model(fs_folder, epoch, fs_network, device)

network = trocarNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
model = trocarTester("trocar", trocar_folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network, path)
model.load_prev(trocar_epoch)
uncorrected_loss[out_joints], corrected_loss[out_joints], torque, fs_pred, pred, jacobian, time = model.test()
wrist_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)
wrist_fs_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)

#############################################
## Load free space and trocar jaw model
#############################################

window = 5
skip = 1
out_joints = [6]
in_joints = [0,1,2,3,4,5]

fs_folder = "free_space_jaw_window"+str(window)+'_'+str(skip)
trocar_folder = "trocar_jaw_2_part_"+str(window) + '_' + str(skip)

fs_network = fsNetwork(window, in_joints=len(in_joints)+1, out_joints=len(out_joints))
fs_network = load_model(fs_folder, epoch, fs_network, device)

network = trocarNetwork(window, len(in_joints)+1, len(out_joints))
model = trocarTester("trocar", trocar_folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network, path, use_jaw=True)
model.load_prev(trocar_epoch)
uncorrected_loss[out_joints], corrected_loss[out_joints], torque, fs_pred, pred, jacobian, time = model.test()
jaw_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)
jaw_fs_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)

#############################################
## Save results and print out
#############################################

path = join('..', 'results', test, 'trocar', contact, exp, 'uncorrected')
try:
    os.mkdir(path)
except OSError:
    print("Result path exists")
    
np.savetxt(join(path, 'arm.csv'), arm_fs_pred)
np.savetxt(join(path, 'insertion.csv'), insertion_fs_pred)
np.savetxt(join(path, 'platform.csv'), platform_fs_pred)
np.savetxt(join(path, 'wrist.csv'), wrist_fs_pred)

path = join('..', 'results', test, 'trocar',  contact, exp,  'corrected')
try:
    os.mkdir(path)
except OSError:
    print("Result path exists")
    
np.savetxt(join(path, 'arm.csv'), arm_pred)
np.savetxt(join(path, 'insertion.csv'), insertion_pred)
np.savetxt(join(path, 'platform.csv'), platform_pred)
np.savetxt(join(path, 'wrist.csv'), wrist_pred)
#np.savetxt(join(path, 'jaw.csv'), jaw_pred)

print('Uncorrected loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (uncorrected_loss[0], uncorrected_loss[1], uncorrected_loss[2], uncorrected_loss[3], uncorrected_loss[4], uncorrected_loss[5], torch.mean(uncorrected_loss)))

print('Corrected loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (corrected_loss[0], corrected_loss[1], corrected_loss[2], corrected_loss[3], corrected_loss[4], corrected_loss[5], torch.mean(corrected_loss)))
