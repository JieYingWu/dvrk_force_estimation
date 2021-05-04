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
net = 'lstm'# sys.argv[1]
seal = 'seal' #sys.argv[2]

JOINTS = 6
window = 1000#utils.WINDOW
skip = 1
root = Path('checkpoints')
in_joints = [0,1,2,3,4,5]
path = '../data/csv/test/trocar/no_contact/'

def main():
    for t in ['60', '120', '180', '240', '300']:#'20', '40', 
        preprocess = 'filtered_torque_' + t + 's' #sys.argv[3]
        dataset = indirectDataset(path, window, skip, in_joints, filter_signal=True, is_rnn=True)
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

        all_time = torch.tensor([])
        all_pred = torch.tensor([])
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

            all_pred = torch.cat((all_pred, step_pred), axis=0) if all_pred.size() else step_pred
            time = time.permute(1,0)
            all_time = torch.cat((all_time, time), axis=0) if all_time.size() else time
                
        all_pred = torch.cat((all_time, all_pred), axis=1)
        results_path = '../results/' + data + '/no_contact/'
        np.savetxt(results_path + '/torque_lstm_troc_' + preprocess + '.csv', all_pred.numpy())
    
if __name__ == "__main__":
    main()


