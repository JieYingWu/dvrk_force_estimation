import sys
import torch
from network import *
import torch.nn as nn
from utils import *
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1000000
fs_epoch = 1000

def make_arm_model(window, skip):
    out_joints = [0,1]
    in_joints = [0,1,2,3,4,5]

    folder = "free_space_arm_window"+str(window) + "_" + str(skip)
    fs_network = armNetwork(window)
    fs_network = load_model(folder, fs_epoch, fs_network, device)

    folder = "trocar_arm_2_part_"+str(window) + '_' + str(skip)
    network = armTrocarNetwork(window)
    model = trocarTester("trocar", folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network)

    return model

def make_insertion_model(window, skip):
    out_joints = [2]
    in_joints = [0,1,2,3,4,5]

    folder = "free_space_insertion_window"+str(window) + "_" + str(skip)
    fs_network = insertionNetwork(window)
    fs_network = load_model(folder, fs_epoch, fs_network, device)

    folder = "trocar_insertion_2_part_"+str(window) + '_' + str(skip)
    network = insertionTrocarNetwork(window)
    model = trocarTester("trocar", folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network)
    return model

def make_wrist_model(window, skip):
    out_joints = [3,4,5]
    in_joints = [3,4,5]

    folder = "free_space_wrist_window"+str(window) + "_" + str(skip)
    fs_network = wristNetwork(window, len(in_joints))
    fs_network = load_model(folder, fs_epoch, fs_network, device)

    folder = "trocar_wrist_2_part_"+str(window) + '_' + str(skip)
    network = wristTrocarNetwork(window, len(in_joints))
    model = trocarTester("trocar", folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network)
    return model

def main():
    joint_name = sys.argv[1]
    window = int(sys.argv[2])
    skip = int(sys.argv[3])
    epoch_to_use = sys.argv[4] 

    if joint_name == "arm":
        model = make_arm_model(window, skip)
    elif joint_name == "insertion":
        model = make_insertion_model(window, skip)
    elif joint_name == "wrist":
        model = make_wrist_model(window, skip)
    else:
        print("Unknown joint name")
        return
    
    print("Loaded a " + joint_name + " model")
    model.load_prev(epoch_to_use)

    uncorrected_loss, corrected_loss, torque, fs_pred, pred = model.test(True)

    all_torque = torch.cat((torque, fs_pred, pred), axis=1).numpy()

    print('Uncorrected loss: ', uncorrected_loss)
    print('Corrected loss: ', corrected_loss)

if __name__ == "__main__":
    main()
