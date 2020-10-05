from os.path import join
import sys
import torch
from network import *
import torch.nn as nn
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 1e-2
batch_size = 4096
epochs = 500
validate_each = 5

def make_arm_model(data, train_path, val_path):
    out_joints = [0,1]
    in_joints = [0,1,2,3,4,5]
    window = 10
    skip = 2
    folder = data + "_arm_window" + str(window) + '_' + str(skip)

    network = fsNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
    model = jointLearner(train_path, val_path, data, folder, network, window, skip, out_joints, in_joints, batch_size, lr, device)
    return model

def make_insertion_model(data, train_path, val_path):
    window = 50
    skip = 2
    out_joints = [2]
    in_joints = [0,1,2,3,4,5]
    folder = data + "_insertion_window" + str(window) + '_' + str(skip)

    network = fsNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
    model = jointLearner(train_path, val_path, data, folder, network, window, skip, out_joints, in_joints, batch_size, lr, device)
    return model

def make_wrist_model(data, train_path, val_path):
    out_joints = [3,4,5]
    in_joints = [0,1,2,3,4,5]
    window = 5
    skip = 1
    folder = data + "_wrist_window" + str(window) + '_' + str(skip)# + "_all_joints"

    network = fsNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
    model = jointLearner(train_path, val_path, data, folder, network, window, skip, out_joints, in_joints, batch_size, lr, device)
    return model

def make_jaw_model(data):
    out_joints = [6]
    in_joints = [0,1,2,3,4,5]
    window = 5
    skip = 1
    folder = data + "_jaw_window" + str(window) + '_' + str(skip)# + "_all_joints"


    train = join('..', 'data', 'csv', 'train_jaw', data)
    val = join('..','data','csv','val_jaw', data)
    network = fsNetwork(window, in_joints=len(in_joints)+1, out_joints=len(out_joints))
    model = jawLearner(train, val, data, folder, network, window, skip, out_joints, in_joints, batch_size, lr, device)
    return model

def main():
    joint_name = sys.argv[1]
    data = sys.argv[2]
    train_path = join('..', 'data', 'csv', 'train', data)
    val_path = join('..','data','csv','val', data)
    
    if joint_name == "arm":
        model = make_arm_model(data, train_path, val_path)
    elif joint_name == "insertion":
        model = make_insertion_model(data, train_path, val_path)
    elif joint_name == "wrist":
        model = make_wrist_model(data, train_path, val_path)
    elif joint_name == "jaw":
        model = make_jaw_model(data)
    else:
        print("Unknown joint name")
        return
    
    print("Loaded a " + data + " model")
    use_previous_model = False
    epoch_to_use = 730

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
