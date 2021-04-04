import sys
import torch
from network import *
import torch.nn as nn
from utils import *
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128
fs_epoch = 1000
contact = 'no_contact' 
window = 100
skip = 1


def main():
    joint = int(sys.argv[1])
    exp = ''#sys.argv[2]
    epoch_to_use = int(sys.argv[2] )
    path = '../data/csv/test/trocar/' + contact + '/' + exp
    in_joints = [0,1,2,3,4,5]

    fs_path = "free_space_lstm" + str(joint)
    fs_network = torqueLstmNetwork(batch_size, device).to(device)
    fs_network = load_model(fs_path, fs_epoch, fs_network, device)

    folder = "trocar_lstm" + str(joint)
    network = trocarNetwork(window, len(in_joints), 1).to(device)

    model = trocarTester(folder, network, window, skip, [joint], in_joints, batch_size, device, fs_network, path, is_rnn=True, use_jaw=False)


    print("Loaded a " + str(joint) + " model")
    model.load_prev(epoch_to_use)

    uncorrected_loss, corrected_loss, torque, fs_pred, pred, jacobian, time = model.test(True)

#    all_torque = torch.cat((torque, fs_pred, pred), axis=1).numpy()

    print('Uncorrected loss: ', uncorrected_loss)
    print('Corrected loss: ', corrected_loss)

if __name__ == "__main__":
    main()
