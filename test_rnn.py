import sys
import torch
from network import *
import torch.nn as nn
from utils import *
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
epoch_to_use = 0
contact = 'with_contact'
window = 5
skip = 1

def main():
    all_pred = None
    for joint in range(6):
        exp = sys.argv[1]
        path = '../data/csv/test/trocar/' + contact + '/' + exp + '/'
        in_joints = [0,1,2,3,4,5]

        folder = "free_space_lstm" + str(joint)
        network = torqueLstmNetwork(batch_size, device).to(device)

        model = jointTester(folder, network, window, skip, [joint], in_joints, batch_size, device, path, is_rnn=True, use_jaw=False)

        print("Loaded a " + str(joint) + " model")
        model.load_prev(epoch_to_use)

        loss, torque, pred, jacobian = model.test(True)
        pred = pred.squeeze(2)
        if all_pred is None:
            all_pred = pred
        else:
            all_pred = torch.cat((all_pred, pred), axis=1)
    np.savetxt(path + '/lstm_pred.csv', all_pred.numpy())
        
#    all_torque = torch.cat((torque, fs_pred, pred), axis=1).numpy()

    print('Loss: ', loss)

if __name__ == "__main__":
    main()
