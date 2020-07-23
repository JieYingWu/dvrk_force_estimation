import torch
import numpy as np
from pathlib import Path
from dataset import directDataset
from network import forceNetwork
from torch.utils.data import DataLoader

IN_CHANNELS = 120
OUT_CHANNELS = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_path = '../data/csv/test/with_contact/'
root = Path('checkpoints' )

batch_size = 4098
epoch_to_use = 4365

network = forceNetwork(IN_CHANNELS, OUT_CHANNELS).to(device)
                        
test_dataset = directDataset(test_path)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=False)
loss_fn = torch.nn.MSELoss()

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

test_loss = np.zeros(3)
for i, (veltorque, sensor) in enumerate(test_loader):
    veltorque = veltorque.to(device)
    sensor = sensor.to(device)

    pred = network(veltorque)
    loss = (pred - sensor)**2
    loss = (loss.sum(0)/pred.size()[0]).sqrt()
    test_loss += loss.cpu().detach().numpy()

print(test_loss/len(test_loader))
