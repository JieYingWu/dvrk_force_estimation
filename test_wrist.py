import sys
import torch
import numpy as np
from pathlib import Path
from dataset import indirectDataset
from network import wristNetwork
from torch.utils.data import DataLoader
from utils import jointTester

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_joints = 3
out_joints = [3,4,5]
in_joints = [3,4,5]
window = int(sys.argv[2])
skip = int(sys.argv[3])

data = sys.argv[1]
folder = data + "_wrist_window"+str(window)+'_'+str(skip)

batch_size = 1000000
epoch_to_use = int(sys.argv[4])

network = wristNetwork(window, len(in_joints))
model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device)
model.load_prev(epoch_to_use)

test_loss = model.test()
        
print('Test loss: t4=%f, t5=%f, t6=%f, mean=%f' % (test_loss[0], test_loss[1], test_loss[2], (torch.mean(test_loss))))
