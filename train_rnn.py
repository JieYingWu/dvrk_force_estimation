from os.path import join
import sys
import torch
from network import *
import torch.nn as nn
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 1e-3
batch_size = 128
epochs = 1000
validate_each = 5
use_previous_model = False
epoch_to_use = 730


def main():
    joint = int(sys.argv[1])
    data = sys.argv[2]

    train_path = join('..', 'data', 'csv', 'train', data)
    val_path = join('..','data','csv','val', data)
    
    in_joints = [0,1,2,3,4,5]
    window = 500
    skip = 1
    
    folder = data + "_lstm" + str(joint)
    
    network = torqueLstmNetwork(batch_size, device)
    model = jointLearner(train_path, val_path, folder, network, window, skip, [joint], in_joints, batch_size, lr, device, is_rnn=True)
    
    print("Loaded a " + data + " model for joint " + str(joint))
        
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

    
if __name__ == "__main__":
    main()
