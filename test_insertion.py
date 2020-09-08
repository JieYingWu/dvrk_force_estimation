import sys
import torch
import numpy as np
from pathlib import Path
from dataset import indirectDataset
from network import insertionNetwork
from torch.utils.data import DataLoader
from utils import jointTester

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
window = int(sys.argv[2])
skip = int(sys.argv[3])
out_joints = [2]
in_joints = [0,1,2,3,4,5]

data = sys.argv[1]
folder = data + "_insertion_window"+str(window)+'_'+str(skip)

batch_size = 1000000
epoch_to_use = int(sys.argv[4])

network = insertionNetwork(window).to(device)
model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device)
model.load_prev(epoch_to_use)

test_loss = model.test()
print('Test loss: f3=%f' % (test_loss[0]))
