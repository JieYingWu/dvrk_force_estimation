import sys
import torch
from network import *
import torch.nn as nn
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1000000
epoch_to_use = 1000

def make_arm_model(data, window, skip):
    out_joints = [0,1]
    in_joints = [0,1,2,3,4,5]
    folder = data + "_arm_window" + str(window) + '_' + str(skip)

    network = armNetwork(window)
    model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device)

    return model

def make_insertion_model(data, window, skip):
    out_joints = [2]
    in_joints = [0,1,2,3,4,5]
    folder = data + "_insertion_window" + str(window) + '_' + str(skip)

    network = insertionNetwork(window)
    model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device)
    return model

def make_wrist_model(data, window, skip):
    out_joints = [3,4,5]
    in_joints = [3,4,5]
    folder = data + "_wrist_window" + str(window) + '_' + str(skip)# + "_all_joints"

    network = wristNetwork(window)
    model = jointLearner(data, folder, network, window, skip, out_joints, in_joints, batch_size, lr, device)
    return model

def main():
    joint_name = sys.argv[1]
    data = sys.argv[2]
    window = int(sys.argv[3])
    skip = int(sys.argv[4])
    epoch_to_use = sys.argv[5] 

    if joint_name == "arm":
        model = make_arm_model(data, window, skip)
    elif joint_name == "insertion":
        model = make_insertion_model(data, window, skip)
    elif joint_name == "wrist":
        model = make_wrist_model(data, window, skip)
    else:
        print("Unknown joint name")
        return
    
    print("Loaded a " + data + " model")
    model.load_prev(epoch_to_use)
    test_loss = model.test()
    print('Test loss: ', test_loss)
    
if __name__ == "__main__":
    main()
