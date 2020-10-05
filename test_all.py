import os
import sys
import torch
import numpy as np
from os.path import join
from dataset import indirectDataset
from network import fsNetwork
from torch.utils.data import DataLoader
from utils import load_model, jointTester

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = sys.argv[1]
contact = sys.argv[2]
epoch_to_use = int(sys.argv[3])
batch_size = 1000000
test_loss = torch.zeros(6)
test_folder = 'test_7dof'
path = join('..', 'data', 'csv', test_folder, data, contact)

#############################################
## Load free space arm model
#############################################

out_joints = [0,1]
in_joints = [0,1,2,3,4,5]
window = 10
skip = 2

folder = data + "_arm_window"+str(window)+'_'+str(skip)

network = fsNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device, path)
model.load_prev(epoch_to_use)
test_loss[0:2], pred, jacobian, time = model.test()
arm_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)
    
#############################################
## Load free space insertion model
#############################################

out_joints = [2]
in_joints = [0,1,2,3,4,5]
window = 50
skip = 1

folder = data + "_insertion_window"+str(window)+'_'+str(skip)

network = fsNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device, path)
model.load_prev(epoch_to_use)
test_loss[2], pred, jacobian, time = model.test()
insertion_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)


#############################################
## Load free space wrist model
#############################################

out_joints = [3,4,5]
in_joints = [0,1,2,3,4,5]
window = 5
skip = 1

folder = data + "_wrist_window"+str(window)+'_'+str(skip)

network = fsNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device, path)
model.load_prev(epoch_to_use)
test_loss[3:], pred, jacobian, time = model.test()
wrist_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)

path = join('..', 'results', test_folder, contact, (data +'_torques'))
try:
    os.mkdir(path)
except OSError:
    print("Result path exists")
    
np.savetxt(join(path, 'arm.csv'), arm_pred)
np.savetxt(join(path, 'insertion.csv'), insertion_pred)
np.savetxt(join(path, 'wrist.csv'), wrist_pred)
print('Test loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4], test_loss[5], (torch.mean(test_loss))))
