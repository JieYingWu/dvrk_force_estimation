import sys
import torch
import numpy as np
from pathlib import Path
from dataset import indirectDataset
from network import armNetwork
from torch.utils.data import DataLoader
from utils import jointTester

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_joints = 2
out_joints = [0,1]
in_joints = [0,1,2,3,4,5]
window = int(sys.argv[2])
skip = int(sys.argv[3])

data = sys.argv[1]
folder = data + "_arm_window"+str(window)+'_'+str(skip)

batch_size = 1000000
epoch_to_use = int(sys.argv[4])

network = armNetwork(window, len(in_joints))
                          
model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device)
model.load_prev(epoch_to_use)

test_loss = model.test()
print('Test loss: t1=%f, t2=%f, mean=%f' % (test_loss[0], test_loss[1], (torch.mean(test_loss))))
