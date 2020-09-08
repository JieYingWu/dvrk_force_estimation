import sys
import torch
from network import wristNetwork
import torch.nn as nn
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
out_joints = [3,4,5]
in_joints = [3,4,5]
window = 2
skip = 1

data = sys.argv[1]
root = Path('checkpoints' ) 
folder = data + "_wrist_window" + str(window) + '_' + str(skip)

lr = 1e-2
batch_size = 4096
epochs = 1000
validate_each = 5
use_previous_model = False
epoch_to_use = 5

network = wristNetwork(window, len(in_joints))
                          
model = jointLearner(data, folder, network, window, skip, out_joints, in_joints, batch_size, lr, device)

if use_previous_model:
    model.load_prev(epoch_to_use)
    epoch = epoch_to_use + 1
else:
    epoch = 1

print('Training for ' + str(epochs))
for e in range(epoch, epochs + 1):

    model.train_step(e)
    
    if e % validate_each == 0:
        model.val_step(e)
