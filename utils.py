from os.path import join
import tqdm
import torch
import torch.nn as nn
from dataset import *
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

max_torque = (torch.tensor([3.1051168, 2.5269854, 8.118658, 0.06744864, 0.1748129, 0.14781484, 0.08568161]))
min_torque = (torch.tensor([-3.1137438, -3.1547165, -9.27689, -0.13118862, -0.16299911, -0.12941329,  -0.08511973]))
range_torque = (max_torque - min_torque)/2 + 0.1*(max_torque - min_torque)
max_torque = [3.32, 3.32, 9.88, 0.344, 0.344, 0.344]

JOINTS = 6
WINDOW = 30
SKIP = 1

def nrmse_loss(y_hat, y, j=0, verbose=False):
    if verbose:
        print(max_torque[j], min_torque[j])
        print(y.max(axis=0).values, y.min(axis=0).values)
        print(y_hat.max(axis=0).values, y_hat.min(axis=0).values)
    denominator = y.max(axis=0).values-y.min(axis=0).values
    rmse = torch.sqrt(torch.mean((y_hat-y)**2, axis=0))
    nrmse = rmse/denominator #range_torque[j] #
    
    return torch.mean(nrmse) * 100

def relELoss(y_hat, y):
    nominator = torch.sum((y_hat-y)**2)
    demonimator = torch.sum(y**2)
    error = torch.sqrt(nominator/denominator)
    return error * 100

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def load_prev(network, model_root, epoch, optimizer=None, scheduler=None):
    if epoch == 0:
        model_path = model_root / 'model_joint_best.pt'
    else:
        model_path = model_root / 'model_joint_{}.pt'.format(epoch)
        
    if model_path.exists():
        state = torch.load(str(model_path))
        network.load_state_dict(state['model'])
        epoch = state['epoch'] + 1
        network.eval()

        if optimizer is not None:
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            network.train()
        print('Restored model, epoch {}'.format(epoch))
    else:
        print('Failed to restore model ' + str(model_path))
        exit()
        
    return epoch
        
def calculate_force(jacobian, joints):
    jacobian = jacobian.numpy()
    force = torch.zeros((joints.shape[0], 6))
    for i in range(joints.shape[0]):
        j = jacobian[i,:].reshape(6,6)
        jacobian_inv_t = torch.from_numpy(np.linalg.inv(j).transpose())
        force[i,:] = torch.matmul(jacobian_inv_t, joints[i,:])
    return force
        
save = lambda ep, model, model_path, error, optimizer, scheduler: torch.save({
    'model': model.state_dict(),
    'epoch': ep,
    'error': error,
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
}, str(model_path))

