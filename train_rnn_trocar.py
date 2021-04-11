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
fs_epoch = 1000
use_previous_model = False
epoch_to_use = 20
window = 5
skip = 1

def main():
    joint = int(sys.argv[1])
    train_path = join('..', 'data', 'csv', 'train', 'trocar')
    val_path = join('..','data','csv','val', 'trocar')

    in_joints = [0,1,2,3,4,5]
            
    fs_path = "lstm/free_space_lstm" + str(joint)
    fs_network = torqueLstmNetwork(batch_size, device).to(device)
    fs_network = load_model(fs_path, fs_epoch, fs_network, device)

    folder = "trocar_lstm" + str(joint)
    network = trocarNetwork(window, len(in_joints), 1).to(device)

    model = trocarLearner(train_path, val_path, folder, network, window, skip, [joint], in_joints, batch_size, lr, device, fs_network, is_rnn=False, filter_signal=False)

    print("Loaded a " + str(joint) + " model")

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
