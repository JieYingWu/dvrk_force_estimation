import sys
import torch
import numpy as np
from pathlib import Path
from dataset import indirectDataset
from network import torqueNetwork
from torch.utils.data import DataLoader
from utils import nrmse_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
JOINTS = 6
window = int(sys.argv[2])
skip = int(sys.argv[3])

data = sys.argv[1]
#test_path = '../data/csv/test/exp1_dynamic_identification/'
test_path = '../data/csv/test/' + data + '/no_contact/'
#test_path = '../data/csv/val/' + data + '/'
root = Path('checkpoints' )
folder = data + "_window"+str(window)+'_'+str(skip)

batch_size = 1000000
epoch_to_use = 1000
loss_fn = torch.nn.MSELoss()

networks = []
for j in range(JOINTS):
    networks.append(torqueNetwork(window))
    networks[j].to(device)
                          

test_dataset = indirectDataset(test_path, window, skip)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=False)

model_root = root / "models_indirect" / folder
for j in range(JOINTS):
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

test_loss = torch.zeros(JOINTS)
    
for i, (position, velocity, torque, jacobian) in enumerate(test_loader):
    position = position.to(device)
    velocity = velocity.to(device)
    posvel = torch.cat((position, velocity), axis=1)
    torque = torque.to(device)

    for j in range(JOINTS):
        pred = networks[j](posvel).detach()
#        loss = loss_fn(pred.squeeze(), torque[:,j])
        loss = nrmse_loss(pred.squeeze(), torque[:,j]).detach()
        test_loss[j] += loss.item()
        
#        pred = pred.transpose(1,0).cpu().numpy()
#        cur_label = torque[:,j].cpu().detach().numpy()
#        cur_torque = np.vstack((pred, cur_label)).transpose()
#        if all_torques is not None:
#            all_torques = np.concatenate((all_torques, cur_torque), axis=1)
#        else:
#            all_torques = cur_torque
print('Test loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4], test_loss[5], (torch.mean(test_loss)/len(test_loader))))
