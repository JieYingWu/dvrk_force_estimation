import sys
import torch
from network import *
import torch.nn as nn
from utils import *
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
fs_epoch = 0
data = 'trocar'
contact = 'no_contact' 
epoch_to_use = int(sys.argv[1])

def main():
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
    all_pred = None

    for i in range(6):
        folder = "lstm/trocar_lstm" + str(i) 
        network = trocarNetwork(window, len(in_joints), 1).to(device)

        model = trocarTester(folder, network, window, skip, [i], in_joints, batch_size, device, path)

        print("Loaded a " + str(i) + " model")
        model.load_prev(epoch_to_use)
        uncorrected_loss, corrected_loss, torque, fs_pred, pred, jacobian, time = model.test(True)

        if all_pred is None:
            all_pred = time.unsqueeze(1)

        all_pred = torch.cat((all_pred, pred), axis=1)
        print('Uncorrected loss: ', uncorrected_loss)
        print('Corrected loss: ', corrected_loss)
    
    np.savetxt('no_contact_pred.csv', all_pred.numpy()) 
    

if __name__ == "__main__":
    main()
