import torch
import numpy as np
from pathlib import Path
from dataset import indirectWindowDataset
from network import torqueWindowNetwork
from torch.utils.data import DataLoader

def nrmse_loss(y_hat, y):
    print(y.max(), y.min())
    print(y_hat.max(), y_hat.min())
    denominator = y.max()-y.min()
    summation = torch.sum((y_hat-y)**2)
    nrmse = torch.sqrt((summation/y.size()[0]))/denominator
    return nrmse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
JOINTS = 6
window = 16

#test_path = '../data/csv/test/exp1_dynamic_identification/'
test_path = '../data/csv/test/no_contact/'
root = Path('checkpoints' )

batch_size = 1000000
epoch_to_use = 1235

networks = []
for j in range(JOINTS):
    networks.append(torqueWindowNetwork(window))
    networks[j].to(device)
                          

test_dataset = indirectWindowDataset(test_path, window)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=False)

model_root = root / "models_indirect"
for j in range(JOINTS):
    model_path = model_root / 'model_joint_{}_{}.pt'.format(j, epoch_to_use)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch'] + 1
        networks[j].load_state_dict(state['model'])
        networks[j].eval()
        print('Restored model, epoch {}, joint {}'.format(epoch-1, j))
    else:
        print('Failed to restore model')
        exit()

test_loss = np.zeros(JOINTS)
    
for i, (posvel, torque, jacobian) in enumerate(test_loader):
    posvel = posvel.to(device)
    torque = torque.to(device)

    step_loss = 0

    all_torques = None
    for j in range(JOINTS):
        pred = networks[j](posvel).detach()
        loss = nrmse_loss(pred.squeeze(), torque[:,j])
        test_loss[j] += loss
        
        pred = pred.transpose(1,0).cpu().numpy()
        cur_label = torque[:,j].cpu().detach().numpy()
        cur_torque = np.vstack((pred, cur_label)).transpose()
        if all_torques is not None:
            all_torques = np.concatenate((all_torques, cur_torque), axis=1)
        else:
            all_torques = cur_torque

test_loss = test_loss*100
print('Test loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f' % (test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4], test_loss[5]))

    