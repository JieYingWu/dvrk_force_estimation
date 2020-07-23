import tqdm
import torch
from pathlib import Path
from dataset import directDataset
from network import forceNetwork
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss import NmrseLoss

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IN_CHANNELS = 120
OUT_CHANNELS = 3

train_path = '../data/csv/train/with_contact/'
val_path = '../data/csv/val/with_contact/'
root = Path('checkpoints' )

lr = 1e-2
batch_size = 4096
epochs = 5000
validate_each = 5
use_previous_model = False
epoch_to_use = 2585

network = forceNetwork(IN_CHANNELS, OUT_CHANNELS).to(device)
optimizer = torch.optim.SGD(network.parameters(), lr)
scheduler = ReduceLROnPlateau(optimizer, verbose=True)
                          
train_dataset = directDataset(train_path)
val_dataset = directDataset(val_path)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size, shuffle=False)
    
loss_fn = torch.nn.MSELoss()

try:
    model_root = root / "models_direct"
    model_root.mkdir(mode=0o777, parents=False)
except OSError:
    print("Model path exists")

# Read existing weights f or both G and D models
if use_previous_model:
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
else:
    init_weights(network)
    epoch = 1

save = lambda ep, model, model_path, error, optimizer, scheduler: torch.save({
    'model': model.state_dict(),
    'epoch': ep,
    'error': error,
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
}, str(model_path))

print('Training for ' + str(epochs))
for e in range(epoch, epochs + 1):

    tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
    tq.set_description('Epoch {}, lr {}'.format(e, optimizer.param_groups[0]['lr']))
    epoch_loss = 0

    network.train()
    
    for i, (veltorque, force) in enumerate(train_loader):
        veltorque = veltorque.to(device)
        force = force.to(device)

        step_loss = 0

        pred = network(veltorque)
        loss = loss_fn(pred.squeeze(), force)
        step_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tq.update(batch_size)
        tq.set_postfix(loss=' loss={:.5f}'.format(step_loss))
        epoch_loss += step_loss

    tq.set_postfix(loss=' loss={:.5f}'.format(epoch_loss/len(train_loader)))
    
    if e % validate_each == 0:
        network.eval()

        val_loss = 0
        for i, (veltorque, force) in enumerate(train_loader):
            veltorque = veltorque.to(device)
            force = force.to(device)
        
            pred = network(veltorque)
            loss = loss_fn(pred.squeeze(), force)
            val_loss += loss.item()
                
        scheduler.step(val_loss)
        tq.set_postfix(loss='validation loss={:5f}'.format(val_loss/len(val_loader)))

        model_path = model_root / "model_{}.pt".format(e)
        save(e, network, model_path, val_loss/len(val_loader), optimizer, scheduler)
        
    tq.close()
