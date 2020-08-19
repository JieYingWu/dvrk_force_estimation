import sys
import torch
import numpy as np
from pathlib import Path
from dataset import indirectWristDataset
from network import wristNetwork
from torch.utils.data import DataLoader
from utils import nrmse_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
JOINTS = 3
joints = [3,4,5]
window = int(sys.argv[2])
skip = int(sys.argv[3])

data = sys.argv[1]
#test_path = '../data/csv/test/exp1_dynamic_identification/'
test_path = '../data/csv/test/' + data + '/no_contact/'
#test_path = '../data/csv/val/' + data + '/'
root = Path('checkpoints' )
folder = data + "_wrist_window"+str(window)+'_'+str(skip)

batch_size = 1000000
epoch_to_use = 1000
loss_fn = torch.nn.MSELoss()

networks = []
for j in range(num_joints):
    networks.append(wristNetwork(window, num_joints))
    networks[j].to(device)
                          

test_dataset = indirectWristDataset(test_path, window, skip, joints)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=False)

model_root = root / "models_indirect" / folder
for j in range(num_joints):
    model_path = model_root / 'model_joint_{}_{}.pt'.format(j, epoch_to_use)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch'] + 1
        networks[j].load_state_dict(state['model'])
        networks[j].eval()
#        print('Restored model, epoch {}, joint {}, window {}, skip {}'.format(epoch-1, j, window, skip))
    else:
        print('Failed to restore model')
        exit()

test_loss = torch.zeros(num_joints)
    
for i, (posvel, torque, jacobian) in enumerate(test_loader):
    posvel = posvel.to(device)
    torque = torque.to(device)

    for j in range(num_joints):
        pred = networks[j](posvel).detach()
        loss = nrmse_loss(pred.squeeze(), torque[:,j]).detach()
        test_loss[j] += loss.item()
        
print('Test loss: t4=%f, t5=%f, t6=%f, mean=%f' % (test_loss[0], test_loss[1], test_loss[2], (torch.mean(test_loss)/len(test_loader))))
