from os.path import join
import sys
import torch
from network import *
import torch.nn as nn
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1000000
epoch_to_use = 0
test = 'test'
contact = 'no_contact'

def make_arm_model(data, path):
    out_joints = [0,1]
    in_joints = [0,1,2,3,4,5]
    window = 10
    skip = 2
    folder = data + "_arm_window" + str(window) + '_' + str(skip)

    network = fsNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
    model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device, path)

    return model

def make_insertion_model(data, path):
    out_joints = [2]
    in_joints = [0,1,2,3,4,5]
    window = 50
    skip = 2
    folder = data + "_insertion_window" + str(window) + '_' + str(skip)

    network = fsNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
    model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device, path)
    return model

def make_platform_model(data, path):
    out_joints = [3]
    in_joints = [0,1,2,3,4,5]
    window = 2
    skip = 1
    folder = data + "_platform_window" + str(window) + '_' + str(skip)# + "_all_joints"

    network = fsNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
    model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device, path)
    return model

def make_wrist_model(data, path):
    out_joints = [4,5]
    in_joints = [0,1,2,3,4,5]
    window = 2
    skip = 1
    folder = data + "_wrist_window" + str(window) + '_' + str(skip)# + "_all_joints"

    network = fsNetwork(window, in_joints=len(in_joints), out_joints=len(out_joints))
    model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device, path)
    return model

def make_jaw_model(data, path):
    out_joints = [6]
    in_joints = [0,1,2,3,4,5]
    window = 5
    skip = 1
    folder = data + "_jaw_window" + str(window) + '_' + str(skip)# + "_all_joints"

    network = fsNetwork(window, in_joints=len(in_joints)+1, out_joints=len(out_joints))
    model = jointTester(data, folder, network, window, skip, out_joints, in_joints, batch_size, device, path, use_jaw=True)
    return model

def main():
    joint_name = sys.argv[1]
    data = sys.argv[2]
    exp = sys.argv[3]
    path = join('..','data','csv', test, data, contact, exp)
    epoch_to_use = sys.argv[4] 

    if joint_name == "arm":
        model = make_arm_model(data, path)
    elif joint_name == "insertion":
        model = make_insertion_model(data, path)
    elif joint_name == "platform":
        model = make_platform_model(data, path)
    elif joint_name == "wrist":
        model = make_wrist_model(data, path)
    elif joint_name == "jaw":
        model = make_jaw_model(data, path)
    else:
        print("Unknown joint name")
        return

    print("Loaded a " + data + " model")
    model.load_prev(epoch_to_use)
    test_loss, pred, jacobian, time = model.test() 
    print('Test loss: ', test_loss)

    results = np.concatenate((time.unsqueeze(1), pred.numpy(), jacobian.numpy()), axis=1)
    path = join('..', 'results',test, contact, data, exp)

    try:
        os.mkdir(path)
    except OSError:
        print("Result path exists")
    np.savetxt(join(path, (joint_name + '.csv')), results)
    
if __name__ == "__main__":
    main()
