import sys
import torch
import numpy as np
from os.path import join, isfile
from dataset import indirectRnnDataset
from network import torqueLstmNetwork
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
JOINTS = 6
exp = 'exp1'

data = sys.argv[1]
#test_path = '../data/csv/test/exp1_dynamic_identification/'
test_path = join('..','data','csv','test', data, 'no_contact', exp)
root = 'checkpoints'
folder = data + "_lstm"

batch_size = 10000000
window = 100
epoch_to_use = int(sys.argv[2])
torque_scale = torch.tensor([4, 4, 10, 0.2, 0.2, 0.2])
loss_fn = torch.nn.MSELoss()


networks = []
for j in range(JOINTS):
    networks.append(torqueLstmNetwork())
    networks[j].to(device)
                          

test_dataset = indirectRnnDataset(test_path, window)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=False)

model_root = join(root, "models_indirect", folder)
for j in range(JOINTS):
    model_path = join(model_root, 'model_joint_{}_{}.pt'.format(j, epoch_to_use))
    if isfile(model_path):
        state = torch.load(str(model_path))
        epoch = state['epoch'] + 1
        networks[j].load_state_dict(state['model'])
        networks[j].eval()
        print('Restored model, epoch {}, joint {}'.format(epoch-1, j))
    else:
        print('Failed to restore model at ' + str(model_path))
        exit()

test_loss = np.zeros(JOINTS)
    
for i, (position, velocity, torque, jacobian) in enumerate(test_loader):
    position = position.to(device)
    velocity = velocity.to(device)
    posvel = torch.cat((position, velocity), axis=2).contiguous()
    torque = torque.to(device)
    posvel = posvel.permute((1,0,2))
    torque = torque.permute((1,0,2))

    step_loss = 0

    all_torques = None
    for j in range(JOINTS):
        pred = networks[j](posvel).detach()*torque_scale[j]
        print(pred.size())
        loss = loss_fn(pred[-1,:,:], torque[-1,:,j:j+1])
        test_loss[j] += loss
        
#        pred = pred.transpose(1,0).cpu().numpy()
#        cur_label = torque[:,j].cpu().detach().numpy()
#        cur_torque = np.vstack((pred, cur_label)).transpose()
#        if all_torques is not None:
#            all_torques = np.concatenate((all_torques, cur_torque), axis=1)
#        else:
#            all_torques = cur_torque

test_loss = test_loss/len(test_loader)
print('Test loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f' % (test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4], test_loss[5]))

    
