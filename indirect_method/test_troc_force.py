import sys
import torch
from network import *
import torch.nn as nn
import utils
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from dataset import indirectDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
data = 'trocar'
epoch_to_use = 0#int(sys.argv[1])
net = 'troc'# sys.argv[1]

JOINTS = utils.JOINTS
window = 1000#utils.WINDOW
skip = 1
root = Path('checkpoints')
in_joints = [0,1,2,3,4,5]
contact = 'no_contact'

def main():
    for t in ['600']:#]:#'20', '40', '120', '240', '360', '480', '600', '720', '840', '960', 
        preprocess = 'filtered_torque_' + t + 's' #sys.argv[3]

        for exp in ['']:#['exp0', 'exp1', 'exp2', 'exp3', 'exp4']:
            path = '../data/csv/test/' + data + '/' + contact + '/' + exp 

            dataset = indirectDataset(path, window, skip, in_joints, filter_signal=False, is_rnn=True)
            loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

            model_root = []
            for j in range(JOINTS):
                folder = "trocar" + str(j)
                model_root.append(root / preprocess / net / folder)

            networks = []
            for j in range(JOINTS):
                networks.append(torqueLstmNetwork(batch_size, device).to(device))
                utils.load_prev(networks[j], model_root[j], epoch_to_use)
                print("Loaded a " + str(j) + " model")

            all_fs_diff = torch.tensor([])
            all_diff = torch.tensor([])
            all_jacobian = torch.tensor([])
            all_time = torch.tensor([])
            uncorrected_loss = 0
            corrected_loss = 0
            loss_fn = torch.nn.MSELoss()

            for i, (position, velocity, torque, jacobian, time) in enumerate(loader):
                position = position.to(device)
                velocity = velocity.to(device)

                step_loss = 0
                step_pred = torch.tensor([])
        
                for j in range(JOINTS):
                    posvel = torch.cat((position, velocity), axis=2).contiguous()                
                    pred = networks[j](posvel).detach().squeeze(0).cpu()
                    step_pred = torch.cat((step_pred, pred), axis=1) if step_pred.size() else pred

                diff = torque.squeeze(0) - step_pred
                all_diff = torch.cat((all_diff, diff), axis=0) if all_diff.size() else diff
                jacobian = jacobian.squeeze(0)
                all_jacobian = torch.cat((all_jacobian, jacobian), axis=0) if all_jacobian.size() else jacobian
                time = time.permute(1,0)
                all_time = torch.cat((all_time, time), axis=0) if all_time.size() else time
                
            all_force = utils.calculate_force(all_jacobian, all_diff)
            all_force = torch.cat((all_time, all_force), axis=1)
            results_path = '../results/' + data + '/' + contact + '/' + exp
            np.savetxt(results_path + '/lstm_troc_' + preprocess + '.csv', all_force.numpy())
    
if __name__ == "__main__":
    main()

