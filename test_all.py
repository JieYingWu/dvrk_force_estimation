import os
import sys
import torch
import numpy as np
from os.path import join
from dataset import indirectDataset
from network import *
from torch.utils.data import DataLoader
from utils import load_model, jointTester

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = sys.argv[1]
contact = sys.argv[2]
exp = sys.argv[3]
epoch_to_use = int(sys.argv[4])
batch_size = 1000000
test_loss = torch.zeros(7)
test_folder = 'test'
path = join('..', 'data', 'csv', test_folder, data, contact, exp)

#############################################
## Load free space arm model
#############################################

out_joints = [0,1]
in_joints = [0,1,2,3,4,5]
window = 10
skip = 2

folder = data + "_arm_window"+str(window)+'_'+str(skip)

network = armNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device, path)
model.load_prev(epoch_to_use)
test_loss[out_joints], pred, jacobian, time = model.test()
arm_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)
    
#############################################
## Load free space insertion model
#############################################

out_joints = [2]
in_joints = [0,1,2,3,4,5]
window = 50
skip = 2

folder = data + "_insertion_window"+str(window)+'_'+str(skip)

network = insertionNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device, path)
model.load_prev(epoch_to_use)
test_loss[out_joints], pred, jacobian, time = model.test()
insertion_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)

#############################################
## Load free space platform model
#############################################

out_joints = [3]
in_joints = [0,1,2,3,4,5]
window = 5
skip = 1

folder = data + "_platform_window"+str(window)+'_'+str(skip)

network = wristNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device, path)
model.load_prev(epoch_to_use)
test_loss[out_joints], pred, jacobian, time = model.test()
platform_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)

#############################################
## Load free space wrist model
#############################################

out_joints = [4,5]
in_joints = [0,1,2,3,4,5]
window = 5
skip = 1

folder = data + "_wrist_window"+str(window)+'_'+str(skip)

network = wristNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device, path)
model.load_prev(epoch_to_use)
test_loss[out_joints], pred, jacobian, time = model.test()
wrist_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)

#############################################
## Load free space jaw model
#############################################

# out_joints = [6]
# in_joints = [0,1,2,3,4,5]
# window = 5
# skip = 1

# folder = data + "_jaw_window"+str(window)+'_'+str(skip)

# network = fsNetwork(window, in_joints=len(in_joints)+1, out_joints=len(out_joints))
# model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device, path, use_jaw=True)
# model.load_prev(epoch_to_use)
# test_loss[out_joints], pred, jacobian, time = model.test()
# jaw_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)


path = join('..', 'results',  test_folder, data, contact, exp)
try:
    os.mkdir(path)
except OSError:
    print("Result path exists")
    
np.savetxt(join(path, 'arm.csv'), arm_pred)
np.savetxt(join(path, 'insertion.csv'), insertion_pred)
np.savetxt(join(path, 'platform.csv'), platform_pred)
np.savetxt(join(path, 'wrist.csv'), wrist_pred)
#np.savetxt(join(path, 'jaw.csv'), jaw_pred)
print('Test loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4], test_loss[5], (torch.mean(test_loss))))
