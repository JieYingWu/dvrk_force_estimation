import sys
import torch
from network import *
import torch.nn as nn
from utils import *
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1000000
fs_epoch = 1000
contact = 'no_contact' # 'no_contact'


def make_arm_model(path):
    out_joints = [0,1]
    in_joints = [0,1,2,3,4,5]
    window = 10
    skip = 2
    
    folder = "free_space_arm_window"+str(window) + "_" + str(skip)
    fs_network = fsNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
    fs_network = load_model(folder, fs_epoch, fs_network, device)

    folder = "trocar_arm_2_part_"+str(window) + '_' + str(skip)
    network = trocarNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
    model = trocarTester("trocar", folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network, path)

    return model

def make_insertion_model(path):
    out_joints = [2]
    in_joints = [0,1,2,3,4,5]
    window = 50
    skip = 2
    
    folder = "free_space_insertion_window"+str(window) + "_" + str(skip)
    fs_network = fsNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
    fs_network = load_model(folder, fs_epoch, fs_network, device)

    folder = "trocar_insertion_2_part_"+str(window) + '_' + str(skip)
    network = trocarNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
    model = trocarTester("trocar", folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network, path)
    return model

def make_platform_model(path):
    out_joints = [3]
    in_joints = [0,1,2,3,4,5]
    window = 2
    skip = 1
    
    folder = "free_space_platform_window"+str(window) + "_" + str(skip)
    fs_network = fsNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
    fs_network = load_model(folder, fs_epoch, fs_network, device)

    folder = "trocar_platform_2_part_"+str(window) + '_' + str(skip)
    network = trocarNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
    model = trocarTester("trocar", folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network, path)
    return model

def make_wrist_model(path):
    out_joints = [4,5]
    in_joints = [0,1,2,3,4,5]
    window = 2
    skip = 1
    
    folder = "free_space_wrist_window"+str(window) + "_" + str(skip)
    fs_network = fsNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
    fs_network = load_model(folder, fs_epoch, fs_network, device)

    folder = "trocar_wrist_2_part_"+str(window) + '_' + str(skip)
    network = trocarNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
    model = trocarTester("trocar", folder, network, window, skip, out_joints, in_joints, batch_size, device, fs_network, path, use_jaw=True)
    return model

def main():
    joint_name = sys.argv[1]
    exp = sys.argv[2]
    epoch_to_use = sys.argv[3] 
    path = '../data/csv/test/trocar/' + contact + '/' + exp

    if joint_name == "arm":
        model = make_arm_model(path)
    elif joint_name == "insertion":
        model = make_insertion_model(path)
    elif joint_name == "platform":
        model = make_platform_model(path)
    elif joint_name == "wrist":
        model = make_wrist_model(path)
    elif joint_name == "jaw":
        model = make_jaw_model(path)
    else:
        print("Unknown joint name")
        return
    
    print("Loaded a " + joint_name + " model")
    model.load_prev(epoch_to_use)

    uncorrected_loss, corrected_loss, torque, fs_pred, pred, jacobian, time = model.test(True)

    all_torque = torch.cat((torque, fs_pred, pred), axis=1).numpy()

    print('Uncorrected loss: ', uncorrected_loss)
    print('Corrected loss: ', corrected_loss)

if __name__ == "__main__":
    main()
