import sys
import torch
import numpy as np
from pathlib import Path
from dataset import indirectDataset
from network import armNetwork, insertionNetwork, wristNetwork, armTrocarNetwork, insertionTrocarNetwork, wristTrocarNetwork
from torch.utils.data import DataLoader
from utils import nrmse_loss, load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_path = '../data/csv/test/trocar/no_contact/'
root = Path('checkpoints' )
batch_size = 1000000
loss_fn = nrmse_loss #torch.nn.MSELoss()
uncorrected_loss = torch.zeros(6)
corrected_loss = torch.zeros(6)
epoch = 1000
trocar_epoch = 500

#############################################
## Load free space and trocar arm model
#############################################

num_joints = 2
window = 10
skip = 2

fs_folder = "free_space_arm_window"+str(window)+'_'+str(skip)
trocar_folder = "trocar_arm_2_part_"+str(window) + '_' + str(skip)

fs_networks= []
networks = []
for j in range(num_joints):
    fs_networks.append(armNetwork(window, 6))
    networks.append(armTrocarNetwork(window))

    fs_networks[j] = load_model(root, fs_folder, epoch, fs_networks[j], j, device)
    networks[j] = load_model(root, trocar_folder, trocar_epoch, networks[j], j, device)

#############################################
## Load dataset and run arm models
#############################################
        
test_dataset = indirectDataset(test_path, window, skip)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=False)

for i, (position, velocity, torque, jacobian) in enumerate(test_loader):
    position = position.to(device)
    velocity = velocity.to(device)
    posvel = torch.cat((position, velocity), axis=1)
    torque = torque.to(device)

    for j in range(num_joints):
        fs_torque = fs_networks[j](posvel).squeeze()
        loss = nrmse_loss(fs_torque, torque[:,j])
        uncorrected_loss[j] += loss.item()

        pred = networks[j](posvel)
        loss = nrmse_loss(pred.squeeze()+fs_torque, torque[:,j])
        corrected_loss[j] += loss.item()


#############################################
## Load free space and trocar insertion model
#############################################

num_joints = 1
window = 50
skip = 2

fs_folder = "free_space_insertion_window"+str(window)+'_'+str(skip)
trocar_folder = "trocar_insertion_2_part_"+str(window) + '_' + str(skip)

fs_network = insertionNetwork(window)
network = insertionTrocarNetwork(window)

fs_network = load_model(root, fs_folder, epoch, fs_network, 0, device)
network = load_model(root, trocar_folder, trocar_epoch, network, 0, device)

#############################################
## Load dataset and run insertion models
#############################################
    
test_dataset = indirectDataset(test_path, window, skip)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=False)

for i, (position, velocity, torque, jacobian) in enumerate(test_loader):
    position = position.to(device)
    velocity = velocity.to(device)
    posvel = torch.cat((position, velocity), axis=1)
    torque = torque.to(device)[:, 2]

    fs_torque = fs_network(posvel).squeeze()
    loss = nrmse_loss(fs_torque, torque)
    uncorrected_loss += loss.item()
    
    pred = network(posvel)
    loss = nrmse_loss(pred.squeeze()+fs_torque, torque)
    corrected_loss += loss.item()


#############################################
## Load free space and trocar wrist model
#############################################

num_joints = 3
window = 20
skip = 5
joints = [3,4,5]

fs_folder = "free_space_wrist_window"+str(window)+'_'+str(skip)
trocar_folder = "trocar_wrist_2_part_"+str(window) + '_' + str(skip)

fs_networks = []
networks = []

for j in range(num_joints):
    fs_networks.append(wristNetwork(window, len(joints)))
    networks.append(wristTrocarNetwork(window, num_joints))

    fs_networks[j] = load_model(root, fs_folder, epoch, fs_networks[j], j, device)
    networks[j] = load_model(root, trocar_folder, trocar_epoch, networks[j], j, device)

#############################################
## Load dataset and run wrist models
#############################################

test_dataset = indirectDataset(test_path, window, skip, joints)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=False)
for i, (position, velocity, torque, jacobian) in enumerate(test_loader):
    position = position.to(device)
    velocity = velocity.to(device)
    posvel = torch.cat((position, velocity), axis=1)
    torque = torque.to(device)

    for j in range(num_joints):
        fs_torque = fs_networks[j](posvel).squeeze()
        loss = nrmse_loss(fs_torque, torque[:,j])
        uncorrected_loss[j] += loss.item()

        pred = networks[j](posvel)
        loss = nrmse_loss(pred.squeeze()+fs_torque, torque[:,j])
        corrected_loss[j] += loss.item()

#############################################
## Save results and print out
#############################################

print('Uncorrected loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (uncorrected_loss[0], uncorrected_loss[1], uncorrected_loss[2], uncorrected_loss[3], uncorrected_loss[4], uncorrected_loss[5], (torch.mean(uncorrected_loss)/len(test_loader))))

print('Corrected loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (corrected_loss[0], corrected_loss[1], corrected_loss[2], corrected_loss[3], corrected_loss[4], corrected_loss[5], (torch.mean(corrected_loss)/len(test_loader))))
