import sys
import torch
from network import *
import torch.nn as nn
import utils
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from dataset import indirectTrocarTestDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128
fs_epoch = 0
data = 'trocar'
contact = 'no_contact'
net = 'ff'
preprocess = 'filtered_torque'
epoch_to_use = int(sys.argv[1])
is_rnn = False

JOINTS = 6
window = 20
skip = 1
root = Path('checkpoints' )

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

    dataset = indirectTrocarTestDataset(path, window, skip, in_joints, is_rnn=is_rnn)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model_root = []
    for j in range(JOINTS):
        folder = "trocar" + str(j)
        model_root.append(root / preprocess / net / folder)

    networks = []
    for j in range(JOINTS):
        networks.append(trocarNetwork(utils.WINDOW, len(in_joints), 1).to(device))
        utils.load_prev(networks[j], model_root[j], epoch_to_use)
        print("Loaded a " + str(j) + " model")

    all_pred = torch.tensor([])
    uncorrected_loss = 0
    corrected_loss = 0
    loss_fn = torch.nn.MSELoss()

    for i, (position, velocity, torque, jacobian, time, fs_pred) in enumerate(loader):
        position = position.to(device)
        velocity = velocity.to(device)
        torque = torque.to(device)
        fs_pred = fs_pred.to(device)

        step_loss = 0
        step_pred = time.unsqueeze(1)
        
        for j in range(JOINTS):
            posvel = torch.cat((position, velocity, fs_pred[:,[j]]), axis=1).contiguous()
            pred = networks[j](posvel) + fs_pred[:,j].unsqueeze(1)
#            pred = networks[j](posvel)*utils.max_torque[j] + fs_pred[:,j].unsqueeze(1)
            loss = loss_fn(pred.squeeze(), torque[:,j])
            step_loss += loss.item()
            step_pred = torch.cat((step_pred, pred.detach().cpu()), axis=1)

        corrected_loss += step_loss / 6
        uncorrected_loss += loss_fn(fs_pred, torque)

        all_pred = torch.cat((all_pred, step_pred), axis=0) if all_pred.size() else step_pred
        
    print('Uncorrected loss: ', uncorrected_loss/len(loader))
    print('Corrected loss: ', corrected_loss/len(loader))
    
    np.savetxt(path + '/no_contact_pred.csv', all_pred.numpy()) 
    

if __name__ == "__main__":
    main()
