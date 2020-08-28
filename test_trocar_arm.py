import sys
import torch
from pathlib import Path
from network import armNetwork, armTrocarNetwork
import torch.nn as nn
from utils import load_model, trocarTester

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
out_joints = [0,1]
num_joints = len(out_joints)
in_joints = [0,1,2,3,4,5]
window = 10
skip = 2
root = Path('checkpoints' ) 

#############################################
## Load free space model
#############################################

epoch_to_use = 1000
folder = "free_space_arm_window"+str(window) + "_" + str(skip)
fs_networks = []
for j in range(num_joints):
    fs_networks.append(armNetwork(window))
    fs_networks[j] = load_model(root, folder, epoch_to_use, fs_networks[j], j, device)


#############################################
## Load trocar model
#############################################

data = "trocar"
test_path = '../data/csv/test/' + data + '/no_contact'
folder = data + "_arm_2_part_"+str(window) + '_' + str(skip)

batch_size = 1000000
epoch_to_use = int(sys.argv[1])

networks = []
for j in range(num_joints):
    networks.append(armTrocarNetwork(window))
    networks[j] = load_model(root, folder, epoch_to_use, networks[j], j, device)

model = trocarTester(data, networks, window, skip, out_joints, in_joints, batch_size, device, fs_networks)
test_loss = model.test()

print('Test loss: t1=%f, t2=%f, mean=%f' % (test_loss[0], test_loss[1], (torch.mean(test_loss))))
