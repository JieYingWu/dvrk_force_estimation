import torch
import numpy as np
from pathlib import Path
from dataset import jointDatasetWithSensor
from network import torqueNetwork
from torch.utils.data import DataLoader

def calculate_force(jacobian, joints):
    force = np.zeros((joints.shape[0], 6))
    for i in range(joints.shape[0]):
        force[i,:] = np.matmul(jacobian[i], joints[i])
    return force

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_path = '../data/csv/test/with_contact/'
root = Path('checkpoints' )

batch_size = 100000
epoch_to_use = 5000

network = torqueNetwork(IN_CHANNELS, OUT_CHANNELS).to(device)

                        
test_dataset = jointDatasetWithSensor(test_path)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=False)

model_path = model_root / 'model_{}.pt'.format(epoch_to_use)
if model_path.exists():
    state = torch.load(str(model_path))
    epoch = state['epoch'] + 1
    network.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    print('Restored model, epoch {}'.format(epoch-1))
else:
    print('Failed to restore model')
    exit()

test_loss = 0
    
for i, (posvel) in enumerate(test_loader):
    posvel = posvel.to(device)

    all_torques = np.zeros((posvel.size()[0], 6))
    for j in range(JOINTS):
        pred = networks[j](posvel).cpu().detach()
        diff_torque = torque[:,j] - pred.transpose(1,0)
        all_torques[:,j] = diff_torque.numpy()        


    pred_force = calculate_force(jacobian, all_torques)
    pred_force = np.concatenate((time.unsqueeze(1), pred_force), axis=1)
np.savetxt('force_sensor_test.csv', pred_force) 
 
    
