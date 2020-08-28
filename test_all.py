import sys
import torch
import numpy as np
from pathlib import Path
from dataset import indirectDataset
from network import armNetwork, insertionNetwork, wristNetwork
from torch.utils.data import DataLoader
from utils import load_model, jointTester

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = sys.argv[1]
root = Path('checkpoints' )
epoch_to_use = 1000
batch_size = 1000000
test_loss = torch.zeros(6)

#############################################
## Load free space arm model
#############################################

out_joints = [0,1]
num_joints = len(out_joints)
in_joints = [0,1,2,3,4,5]
window = 10
skip = 2

folder = data + "_arm_window"+str(window)+'_'+str(skip)

arm_networks = []
for j in range(num_joints):
    arm_networks.append(armNetwork(window))
    arm_networks[j] = load_model(root, folder, epoch_to_use, arm_networks[j], j, device)

model = jointTester(data, arm_networks, window, skip, out_joints, in_joints, batch_size, device)
test_loss[0:2] = model.test()
    
#############################################
## Load free space insertion model
#############################################

out_joints = [2]
num_joints = len(out_joints)
in_joints = [0,1,2,3,4,5]
window = 50
skip = 2

folder = data + "_insertion_window"+str(window)+'_'+str(skip)

insertion_networks = []
for j in range(num_joints):
    insertion_networks.append(insertionNetwork(window))
    insertion_networks[j] = load_model(root, folder, epoch_to_use, insertion_networks[j], j, device)

model = jointTester(data, insertion_networks, window, skip, out_joints, in_joints, batch_size, device)
test_loss[2] = model.test()

#############################################
## Load free space wrist model
#############################################

out_joints = [3,4,5]
num_joints = len(out_joints)
in_joints = [3,4,5]
window = 20
skip = 5

folder = data + "_wrist_window"+str(window)+'_'+str(skip)

wrist_networks = []
for j in range(num_joints):
    wrist_networks.append(wristNetwork(window, 3))
    wrist_networks[j] = load_model(root, folder, epoch_to_use, wrist_networks[j], j, device)
        
model = jointTester(data, wrist_networks, window, skip, out_joints, in_joints, batch_size, device)
test_loss[3:] = model.test()

print('Test loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4], test_loss[5], (torch.mean(test_loss))))
