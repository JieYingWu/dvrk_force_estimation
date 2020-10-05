import sys
import torch
from network import *
import torch.nn as nn
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 1e-2
batch_size = 4096
epochs = 1000
validate_each = 5
fs_epoch = 1000
    
def make_arm_model(train_path, val_path):
    out_joints = [0,1]
    in_joints = [0,1,2,3,4,5]
    window = 10
    skip = 2
    
    folder = "free_space_arm_window"+str(window) + "_" + str(skip)
    fs_network = fsNetwork(window, len(in_joints), len(out_joints))
    fs_network = load_model(folder, fs_epoch, fs_network, device)

    folder = "trocar_arm_2_part_"+str(window) + '_' + str(skip)
    network = trocarNetwork(window, len(in_joints), len(out_joints))
    model = trocarLearner(train_path, val_path, "trocar", folder, network, window, skip, out_joints, in_joints, batch_size, lr, device, fs_network)
    return model

def make_insertion_model(train_path, val_path):
    window = 50
    skip = 2
    out_joints = [2]
    in_joints = [0,1,2,3,4,5]
    
    folder = "free_space_insertion_window"+str(window) + "_" + str(skip)
    fs_network = fsNetwork(window, len(in_joints), len(out_joints))
    fs_network = load_model(folder, fs_epoch, fs_network, device)

    folder = "trocar_insertion_2_part_"+str(window) + '_' + str(skip)
    network = trocarNetwork(window, len(in_joints), len(out_joints))
    model = trocarLearner(train_path, val_path, "trocar", folder, network, window, skip, out_joints, in_joints, batch_size, lr, device, fs_network)
    return model

def make_platform_model(train_path, val_path):
    out_joints = [3]
    in_joints = [0,1,2,3,4,5]
    window = 5
    skip = 1

    folder = "free_space_platform_window"+str(window) + "_" + str(skip)
    fs_network = fsNetwork(window, len(in_joints), len(out_joints))
    fs_network = load_model(folder, fs_epoch, fs_network, device)

    folder = "trocar_platform_2_part_"+str(window) + '_' + str(skip)
    network = trocarNetwork(window, len(in_joints), len(out_joints))
    model = trocarLearner(train_path, val_path, "trocar", folder, network, window, skip, out_joints, in_joints, batch_size, lr, device, fs_network)
    return model

def make_wrist_model(train_path, val_path):
    out_joints = [4,5]
    in_joints = [0,1,2,3,4,5]
    window = 5
    skip = 1

    folder = "free_space_wrist_window"+str(window) + "_" + str(skip)
    fs_network = fsNetwork(window, len(in_joints), len(out_joints))
    fs_network = load_model(folder, fs_epoch, fs_network, device)

    folder = "trocar_wrist_2_part_"+str(window) + '_' + str(skip)
    network = trocarNetwork(window, len(in_joints), len(out_joints))
    model = trocarLearner(train_path, val_path, "trocar", folder, network, window, skip, out_joints, in_joints, batch_size, lr, device, fs_network)
    return model

def main():
    joint_name = sys.argv[1]
    train_path = join('..', 'data', 'csv', 'train', 'trocar')
    val_path = join('..','data','csv','val', 'trocar')
    
    if joint_name == "arm":
        model = make_arm_model(train_path, val_path)
    elif joint_name == "insertion":
        model = make_insertion_model(train_path, val_path)
    elif joint_name == "platform":
        model = make_platform_model(train_path, val_path)
    elif joint_name == "wrist":
        model = make_wrist_model(train_path, val_path)
    else:
        print("Unknown joint name")
        return
    
    print("Loaded a " + joint_name + " model")
    use_previous_model = False
    epoch_to_use = 20

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
