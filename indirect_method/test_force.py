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
epoch_to_use = 0#int(sys.argv[1])
net = sys.argv[1]
seal = sys.argv[2]

JOINTS = utils.JOINTS
window = utils.WINDOW
skip = utils.SKIP
root = Path('checkpoints' )
in_joints = [0,1,2,3,4,5]
contact = 'no_contact'

def main():
    for t in ['600']:#'120', '240', '360', '480', '600', '720', '840', '960', '1080']:#'20', '40', 
        preprocess = 'filtered_torque_' + t + 's' #sys.argv[3]

        for exp in ['']:#['exp0', 'exp1', 'exp2', 'exp3', 'exp4']:#
            path = '../data/csv/test/' + data + '/' + contact + '/' + exp 

            dataset = indirectTrocarTestDataset(path, window, skip, in_joints, seal=seal, filter_signal=False, net=net)
            loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

            model_root = []
            for j in range(JOINTS):
                folder = "trocar" + str(j)
                model_root.append(root / preprocess / net / folder)

            networks = []
            for j in range(JOINTS):
                networks.append(trocarNetwork(window, len(in_joints), 1).to(device))
                utils.load_prev(networks[j], model_root[j], epoch_to_use)
                print("Loaded a " + str(j) + " model")

            all_fs_diff = torch.tensor([]) 
            all_diff = torch.tensor([])
            all_jacobian = torch.tensor([])
            all_time = torch.tensor([])
            uncorrected_loss = 0
            corrected_loss = 0
            loss_fn = torch.nn.MSELoss()

            for i, (position, velocity, torque, jacobian, time, fs_pred) in enumerate(loader):
                position = position.to(device)
                velocity = velocity.to(device)

                step_loss = 0
                step_pred = torch.tensor([])
        
                for j in range(JOINTS):
                    posvel = torch.cat((position, velocity, fs_pred[:,[j]].to(device)), axis=1).contiguous()                
                    pred = networks[j](posvel).detach().cpu() + fs_pred[:,[j]]
                    step_pred = torch.cat((step_pred, pred), axis=1) if step_pred.size() else pred

                fs_pred = fs_pred[:,[j]]
                fs_diff = torque - fs_pred
                diff = torque - step_pred
                all_fs_diff = torch.cat((all_fs_diff, fs_diff), axis=0) if all_fs_diff.size() else fs_diff
                all_diff = torch.cat((all_diff, diff), axis=0) if all_diff.size() else diff
                all_jacobian = torch.cat((all_jacobian, jacobian), axis=0) if all_jacobian.size() else jacobian
                all_time = torch.cat((all_time, time), axis=0) if all_time.size() else time

            all_time = all_time.unsqueeze(1)
            all_fs_force = utils.calculate_force(all_jacobian, all_fs_diff)
            all_fs_force = torch.cat((all_time, all_fs_force), axis=1)
            results_path = '../results/' + data + '/' + contact + '/' + exp 
            np.savetxt(results_path + '/uncorrected_' + net + '_' + seal + '_' + preprocess + '.csv', all_fs_force.numpy()) 

            all_force = utils.calculate_force(all_jacobian, all_diff)
            all_force = torch.cat((all_time, all_force), axis=1)
            np.savetxt(results_path + '/corrected_' + net + '_' + seal + '_' + preprocess + '.csv', all_force.numpy())
    
if __name__ == "__main__":
    main()
