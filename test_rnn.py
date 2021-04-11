import sys
import torch
from network import *
import torch.nn as nn
from utils import *
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
epoch_to_use = 0
contact = 'no_contact'
window = 20
skip = 1
is_rnn = True

def main():
    all_pred = None
    for joint in range(6):
        exp = sys.argv[1]
        path = '../data/csv/test/trocar/' + contact + '/'# + exp + '/'
        in_joints = [0,1,2,3,4,5]

        if is_rnn:
            folder = "lstm/free_space_lstm" + str(joint) 
            network = torqueLstmNetwork(batch_size, device).to(device)
        else:
            folder = "free_space" + str(joint)
            network = fsNetwork(window)

        model = jointTester(folder, network, window, skip, [joint], in_joints, batch_size, device, path, is_rnn=is_rnn, use_jaw=False)

        print("Loaded a " + str(joint) + " model")
        model.load_prev(epoch_to_use)

        loss, torque, pred, jacobian, time = model.test(True)

        if is_rnn:
            pred = pred.squeeze(2)
        
        if all_pred is None:
            if is_rnn:
                all_pred = time
            else:
                all_pred = time.unsqueeze(1)

        all_pred = torch.cat((all_pred, pred), axis=1)

    if is_rnn:
        np.savetxt(path + 'lstm_pred.csv', all_pred.numpy())
    else:
        np.savetxt(path + 'ff_pred.csv', all_pred.numpy())
        
#    all_torque = torch.cat((torque, fs_pred, pred), axis=1).numpy()

    print('Loss: ', loss)

if __name__ == "__main__":
    main()
