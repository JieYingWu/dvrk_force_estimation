import sys
import torch
from network import *
import torch.nn as nn
from utils import *
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
contact = 'no_contact'
is_rnn = True
data = 'free_space'

def main():
    all_pred = None
    epoch_to_use = int(sys.argv[1])
    exp = sys.argv[2]
    if exp == 'train':
        path = '../data/csv/train/' + data + '/'
    elif exp == 'val':
        path = '../data/csv/val/' + data + '/'
    elif exp =='test':
        path = '../data/csv/test/' + data + '/' + contact + '/'
    else:
        path = '../data/csv/test/' + data + '/' + contact + '/' + exp + '/'
    in_joints = [0,1,2,3,4,5]
    
    for joint in range(6):
        if is_rnn:
            window = 1000
            folder = "lstm/free_space" + str(joint) 
            network = torqueLstmNetwork(batch_size, device).to(device)
        else:
            folder = "ff/free_space" + str(joint)
            network = fsNetwork(window)
        
        model = jointTester(folder, network, window, skip, [joint], in_joints, batch_size, device, path, is_rnn=is_rnn)

        print("Loaded a " + str(joint) + " model")
        model.load_prev(epoch_to_use)

        loss, torque, pred, jacobian, time = model.test(True)
        
        if all_pred is None:
            all_pred = time.unsqueeze(1)

        all_pred = torch.cat((all_pred, pred.unsqueeze(1)), axis=1)
        print(all_pred.size())

    if is_rnn:
        np.savetxt(path + 'lstm_pred.csv', all_pred.numpy())
    else:
        np.savetxt(path + 'ff_pred.csv', all_pred.numpy())
        
#    all_torque = torch.cat((torque, fs_pred, pred), axis=1).numpy()

    print('Loss: ', loss)

if __name__ == "__main__":
    main()
