import tqdm
import torch
from pathlib import Path
from dataset import jointDataset
from network import torqueNetwork
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
JOINTS = 6

train_path = '../data/csv/train/'
val_path = '../data/csv/val/'
root = Path('checkpoints' )

lr = 1e-3
batch_size = 2048
epochs = 500
validate_each = 5
use_previous_model = True
epoch_to_use = 60

networks = []
optimizers = []
for j in range(JOINTS):
    networks.append(torqueNetwork())
    networks[j].to(device)
    optimizers.append(torch.optim.Adam(networks[j].parameters(), lr))
                          

train_dataset = jointDataset(train_path)
val_dataset = jointDataset(val_path)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size, shuffle=False)
    
loss_fn = torch.nn.MSELoss()

try:
    model_root = root / "models"
    model_root.mkdir(mode=0o777, parents=False)
except OSError:
    print("Model path exists")

# Read existing weights for both G and D models
if use_previous_model:
    for j in range(JOINTS):
        model_path = model_root / 'model_joint_{}_{}.pt'.format(j, epoch_to_use)
        if model_path.exists():
            networks[j].load_state_dict(state['model'])
            optimizers[j].load_state_dict(state['optimizer'])
            print('Restored model, epoch {}, joint {}'.format(epoch-1, j))
        else:
            print('Failed to restore model')
            exit()
        state = torch.load(str(model_path))
        epoch = state['epoch'] + 1
else:
    epoch = 1

save = lambda ep, model, model_path, error, optimizer: torch.save({
    'model': model.state_dict(),
    'epoch': ep,
    'error': error,
    'optimizer': optimizer.state_dict(),
}, str(model_path))

print('Training for ' + str(epochs))
for e in range(epoch, epochs + 1):

    tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
    tq.set_description('Epoch {}, lr {}'.format(e, lr))
    epoch_loss = 0
    
    for i, (posvel, torque) in enumerate(train_loader):
        posvel = posvel.to(device)
        torque = torque.to(device)

        step_loss = 0

        for j in range(JOINTS):
            pred = networks[j](posvel)
            loss = loss_fn(pred.squeeze(), torque[:,j])
            step_loss += loss.item()
            optimizers[j].zero_grad()
            loss.backward()
            optimizers[j].step()

        tq.update(batch_size)
        tq.set_postfix(loss=' loss={:.5f}'.format(step_loss))
        epoch_loss += step_loss

    tq.set_postfix(loss=' loss={:.5f}'.format(epoch_loss/len(train_loader)))
    
    if e % validate_each == 0:
        val_loss = 0
        for i, (posvel, torque) in enumerate(train_loader):
            posvel = posvel.to(device)
            torque = torque.to(device)

            step_loss = 0
            
            for j in range(JOINTS):
                pred = networks[j](posvel)
                loss = loss_fn(pred.squeeze(), torque[:,j])
                step_loss += loss.item()

            val_loss += step_loss
        tq.set_postfix(loss='validation loss={:5f}'.format(val_loss/len(val_loader)))

        for j in range(JOINTS):
            model_path = model_root / "model_joint_{}_{}.pt".format(j, e)
            save(e, networks[j], model_path, val_loss/len(val_loader), optimizers[j])
        
    tq.close()
