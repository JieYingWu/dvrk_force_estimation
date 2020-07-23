import torch
import numpy as np
from pathlib import Path
from dataset import directDataset
from network import forceNetwork
from torch.utils.data import DataLoader

IN_CHANNELS = 120
OUT_CHANNELS = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

phantom = 'a'
test_path = '../data/csv/test/phantoms/' + phantom + '/'
root = Path('checkpoints' )

batch_size = 10000000
epoch_to_use = 5000

network = forceNetwork(IN_CHANNELS, OUT_CHANNELS).to(device)
                        
test_dataset = directDataset(test_path)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=False)

model_root = root / "models_direct"
model_path = model_root / 'model_{}.pt'.format(epoch_to_use)
if model_path.exists():
    state = torch.load(str(model_path))
    epoch = state['epoch'] + 1
    network.load_state_dict(state['model'])
    network.eval()
    print('Restored model, epoch {}'.format(epoch-1))
else:
    print('Failed to restore model')
    exit()

for i, (veltorque, sensor, cartesian) in enumerate(test_loader):
    veltorque = veltorque.to(device)
    pred = network(veltorque)
    pred = np.concatenate((cartesian, pred.cpu().detach().numpy()), axis=1)

np.savetxt('phantom_test_direct_' + phantom + '.csv', pred) 
