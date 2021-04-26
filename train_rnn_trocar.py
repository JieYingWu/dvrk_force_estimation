import sys
import torch
from os.path import join
from network import *
import torch.nn as nn
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 1e-3
batch_size = 128
epochs = 1000
validate_each = 5
fs_epoch = 0
use_previous_model = False
epoch_to_use = 20
is_rnn = False

def main():
    joint = int(sys.argv[1])
    train_path = join('..', 'data', 'csv', 'train', 'trocar')
    val_path = join('..','data','csv','val', 'trocar')

    in_joints = [0,1,2,3,4,5]
            
    folder = "trocar" + str(joint)
    network = trocarNetwork(utils.WINDOW, len(in_joints), 1).to(device)

    model = utils.trocarLearner(train_path, val_path, folder, network, utils.WINDOW, utils.SKIP, [joint], in_joints, batch_size, lr, device, is_rnn=is_rnn, filter_signal=False)

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
