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
data = 'trocar'
contact = 'no_contact'
net = sys.argv[1]
epoch_to_use = 0#int(sys.argv[1])
seal = sys.argv[2]
is_seal = seal == 'seal'

JOINTS = utils.JOINTS
root = Path('checkpoints' )

def main():
    path = '../data/csv/test/' + data + '/' + contact + '/'
    in_joints = [0,1,2,3,4,5]

    dataset = indirectTrocarTestDataset(path, utils.WINDOW, utils.SKIP, in_joints, seal=seal, net=net)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    for t in ['60', '120', '180', '240', '300']:
        preprocess = 'filtered_torque_' + t + 's'
        
        model_root = []
        for j in range(JOINTS):
            if is_seal:
                folder = "trocar" + str(j)
            else:
                folder = "trocar_no_cannula" + str(j)
            model_root.append(root / preprocess / net / folder)

        networks = []
        for j in range(JOINTS):
            networks.append(trocarNetwork(utils.WINDOW, len(in_joints)).to(device))
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
                pred = networks[j](posvel) + fs_pred[:,[j]]
#            pred = networks[j](posvel)*utils.max_torque[j] + fs_pred[:,j].unsqueeze(1)
                loss = loss_fn(pred.squeeze(), torque[:,j])
                step_loss += loss.item()
                step_pred = torch.cat((step_pred, pred.detach().cpu()), axis=1)

            corrected_loss += step_loss / 6
            in_joints = np.array(in_joints)
#        fs_pred = fs_pred[:,(in_joints+1)*utils.WINDOW-1]
            uncorrected_loss += loss_fn(fs_pred, torque)

            all_pred = torch.cat((all_pred, step_pred), axis=0) if all_pred.size() else step_pred
        
        print('Uncorrected loss: ', uncorrected_loss/len(loader))
        print('Corrected loss: ', corrected_loss/len(loader))
        np.savetxt('../results/' + data + '/' + contact + '/torque_' + net + '_' + seal + '_' + preprocess + '.csv', all_pred.numpy()) 
    

if __name__ == "__main__":
    main()
