import sys
import torch
import numpy as np
from os.path import join, isfile
from dataset import indirectRnnDataset
from network import torqueLstmNetwork
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1000000
uncorrected_loss = torch.zeros(6)
corrected_loss = torch.zeros(6)
epoch = int(sys.argv[1])

out_joints = [0]
in_joints = [0,1,2,3,4,5]

fs_folder = 
fs_network = torqueLstmNetwork()
fs_network = load_model(fs_folder, epoch, fs_network, device)

model = forceTrocarTester("trocar", trocar_folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network)

model.load_prev(trocar_epoch)
uncorrected_loss[0:2], corrected_loss[0:2], torque, fs_pred, pred, jacobian, time = model.test()
arm_fs_pred = np.concatenate((time.unsqueeze(1), fs_pred.numpy(), jacobian.numpy()), axis=1)
arm_pred = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)


test_loss = test_loss/len(test_loader)
print('Test loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f' % (test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4], test_loss[5]))

    
