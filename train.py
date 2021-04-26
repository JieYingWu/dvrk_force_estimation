import sys
import tqdm
import torch
from pathlib import Path
from dataset import indirectDataset
from network import torqueLstmNetwork, fsNetwork
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import init_weights, WINDOW, range_torque, save, load_prev
from os.path import join

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
JOINTS = 6
skip = 1

data = sys.argv[1]
train_path = join('..', 'data', 'csv', 'train', data)
val_path = join('..','data','csv','val', data)
root = Path('checkpoints' )
is_rnn = bool(int(sys.argv[2]))
if is_rnn:
    folder = "lstm/free_space"
else:
    folder = "ff/free_space"

lr = 1e-2
batch_size = 256
epochs = 1000
validate_each = 5
use_previous_model = False
epoch_to_use = 10
in_joints = [0,1,2,3,4,5]
f = False
print('Running for is_rnn value: ', is_rnn)

networks = []
optimizers = []
schedulers = []
model_root = []

for j in range(JOINTS):
    if is_rnn:
        window = 1000
        networks.append(torqueLstmNetwork(batch_size, device))
    else:
        window = WINDOW
        networks.append(fsNetwork(window))

    networks[j].to(device)
    optimizers.append(torch.optim.SGD(networks[j].parameters(), lr))
    schedulers.append(ReduceLROnPlateau(optimizers[j], verbose=True))
                          
train_dataset = indirectDataset(train_path, window, skip, in_joints, is_rnn=is_rnn, filter_signal=f)
val_dataset = indirectDataset(val_path, window, skip, in_joints, is_rnn=is_rnn, filter_signal=f)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size, shuffle=False, drop_last=True)
    
loss_fn = torch.nn.MSELoss()

for j in range(JOINTS):
    try:
        model_root.append(root / "unfiltered" / (folder + str(j)))
        model_root[j].mkdir(mode=0o777, parents=False)
    except OSError:
        print("Model path exists")

# Read existing weights for both G and D models
if use_previous_model:
    for j in range(JOINTS):
        epoch = load_prev(networks[j], model_root[j], epoch_to_use, optimizers[j], schedulers[j])
else:
    for j in range(JOINTS):
        init_weights(networks[j])
    epoch = 1

print('Training for ' + str(epochs))
best_loss = torch.zeros(6) + 1e8

for e in range(epoch, epochs + 1):

    tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
    tq.set_description('Epoch {}, lr {}'.format(e, optimizers[0].param_groups[0]['lr']))
    epoch_loss = 0

    for j in range(JOINTS):
        networks[j].train()
    
    for i, (position, velocity, torque, jacobian, time) in enumerate(train_loader):
        position = position.to(device)
        velocity = velocity.to(device)
        torque = torque.to(device)
        if is_rnn: 
            posvel = torch.cat((position, velocity), axis=2).contiguous()
        else:
            posvel = torch.cat((position, velocity), axis=1).contiguous()

        step_loss = 0

        for j in range(JOINTS):
            pred = networks[j](posvel) * range_torque[j]
            if is_rnn:
                loss = loss_fn(pred.squeeze(), torque[:,:,j])
            else:
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
        for j in range(JOINTS):
            networks[j].eval()

        val_loss = torch.zeros(JOINTS)
        for i, (position, velocity, torque, jacobian, time) in enumerate(val_loader):
            position = position.to(device)
            velocity = velocity.to(device)
            torque = torque.to(device)
            if is_rnn: 
                posvel = torch.cat((position, velocity), axis=2).contiguous()
            else:
                posvel = torch.cat((position, velocity), axis=1).contiguous()

            for j in range(JOINTS):
                pred = networks[j](posvel) * range_torque[j]
                if is_rnn:
                    loss = loss_fn(pred.squeeze(), torque[:,:,j])
                else:
                    loss = loss_fn(pred.squeeze(), torque[:,j])
                val_loss[j] += loss.item()

            val_loss = val_loss / len(val_loader)
                
        for j in range(JOINTS):
            schedulers[j].step(val_loss[j])
        tq.set_postfix(loss='validation loss={:5f}'.format(torch.mean(val_loss)))

        for j in range(JOINTS):
            model_path = model_root[j] / "model_joint_{}.pt".format(e)
            save(e, networks[j], model_path, val_loss[j], optimizers[j], schedulers[j])

            if val_loss[j] < best_loss[j]:
                model_path = model_root[j] / "model_joint_best.pt"
                save(e, networks[j], model_path, val_loss[j], optimizers[j], schedulers[j])
                best_loss[j] = val_loss[j]

        
    tq.close()
