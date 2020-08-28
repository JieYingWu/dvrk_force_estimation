import sys
import torch
import numpy as np
from pathlib import Path
from dataset import indirectDataset
from network import armNetwork
from torch.utils.data import DataLoader
from utils import nrmse_loss, load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_joints = 2
joints = [0,1,2,3,4,5]
window = int(sys.argv[2])
skip = int(sys.argv[3])

data = sys.argv[1]
#test_path = '../data/csv/test/exp1_dynamic_identification/'
test_path = '../data/csv/test/' + data + '/no_contact/'
#test_path = '../data/csv/val/' + data + '/'
root = Path('checkpoints' )
folder = data + "_arm_window"+str(window)+'_'+str(skip)

batch_size = 1000000
epoch_to_use = int(sys.argv[4])
loss_fn = torch.nn.MSELoss()

networks = []
for j in range(num_joints):
    networks.append(armNetwork(window, len(joints)))
    networks[j] = load_model(root, folder, epoch_to_use, networks[j], j, device)
                          
test_dataset = indirectDataset(test_path, window, skip, joints)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=False)
test_loss = torch.zeros(num_joints)
        
for i, (position, velocity, torque, jacobian) in enumerate(test_loader):
    position = position.to(device)
    velocity = velocity.to(device)
    posvel = torch.cat((position, velocity), axis=1)
    torque = torque.to(device)[:,0:2]

    for j in range(num_joints):
        pred = networks[j](posvel).detach()
        loss = nrmse_loss(pred.squeeze(), torque[:,j]).detach()
        test_loss[j] += loss.item()
        
print('Test loss: t1=%f, t2=%f, mean=%f' % (test_loss[0], test_loss[1], (torch.mean(test_loss)/len(test_loader))))
