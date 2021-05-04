import sys
import tqdm
import torch
from pathlib import Path
from dataset import indirectTrocarDataset
from network import trocarNetwork
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import init_weights, JOINTS, WINDOW, SKIP, save, load_prev, max_torque
from os.path import join

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = 'trocar'
train_path = join('..', 'data', 'csv', 'train', data)
val_path = join('..','data','csv','val', data)
root = Path('checkpoints' )
net = sys.argv[1]
seal = sys.argv[2]
is_seal = 'seal' == seal
if is_seal:
    folder = net + '/trocar'
else:
    folder = net + '/trocar_no_cannula'

lr = 1e-4
batch_size = 128
epochs = 400
validate_each = 5
use_previous_model = False
epoch_to_use = 10
in_joints = [0,1,2,3,4,5]
f = True
print('Running for is_seal value: ', is_seal)
loss_fn = torch.nn.MSELoss()

    
for num in ['60', '120', '180', '240', '300', '360']:
    model = 'filtered_torque_' + num + 's'
    n = int(num)

    try:
        temp = root / model 
        temp.mkdir(mode=0o777, parents=False)
    except OSError:
        print("Model path exists")

    try:
        temp = root / model / net
        temp.mkdir(mode=0o777, parents=False)
    except OSError:
        print("Net path exists")
    
    networks = []
    optimizers = []
    schedulers = []
    model_root = []

    for j in range(JOINTS):
        networks.append(trocarNetwork(WINDOW, len(in_joints), 1))
        networks[j].to(device)
        optimizers.append(torch.optim.Adam(networks[j].parameters(), lr))
        schedulers.append(ReduceLROnPlateau(optimizers[j], verbose=True))
                          
    train_dataset = indirectTrocarDataset(train_path, WINDOW, SKIP, in_joints, num=n, seal=seal, filter_signal=f, net=net)
    val_dataset = indirectTrocarDataset(val_path, WINDOW, SKIP, in_joints, num=n, seal=seal, filter_signal=f, net=net)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    for j in range(JOINTS):
        try:
            model_root.append(root / model / (folder + str(j)))
            model_root[j].mkdir(mode=0o777, parents=False)
        except OSError:
            print("Model path exists")
 
    if use_previous_model:
        for j in range(JOINTS):
            epoch = load_prev(networks[j], model_root[j], epoch_to_use, optimizers[j], schedulers[j])
    else:
        for j in range(JOINTS):
            init_weights(networks[j])
        epoch = 1

    print('Training for ' + str(epochs))
    best_loss = torch.zeros(6) + 1e10

    for e in range(epoch, epochs + 1):

        tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
        tq.set_description('Epoch {}, lr {}'.format(e, optimizers[0].param_groups[0]['lr']))
        epoch_loss = 0

        for j in range(JOINTS):
            networks[j].train()
    
        for i, (position, velocity, torque, time, fs_pred) in enumerate(train_loader):
            position = position.to(device)
            velocity = velocity.to(device)
            fs_pred = fs_pred.to(device)
            torque = torque.to(device)
            step_loss = 0

            for j in range(JOINTS):
                posvel = torch.cat((position, velocity, fs_pred[:,[j]]), axis=1).contiguous()
                pred = networks[j](posvel) + fs_pred[:,[j]]
        
                loss = loss_fn(pred, torque[:,[j]])
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
            for i, (position, velocity, torque, time, fs_pred) in enumerate(val_loader):
                position = position.to(device)
                velocity = velocity.to(device)
                fs_pred = fs_pred.to(device)
                torque = torque.to(device)

                for j in range(JOINTS):
                    posvel = torch.cat((position, velocity, fs_pred[:,[j]]), axis=1).contiguous()
                    pred = networks[j](posvel) + fs_pred[:,[j]]
                    loss = loss_fn(pred, torque[:,[j]])
                    val_loss[j] += loss.item()

            val_loss = val_loss / len(val_loader)
                
            for j in range(JOINTS):
                schedulers[j].step(val_loss[j])
                model_path = model_root[j] / "model_joint_{}.pt".format(e)
                save(e, networks[j], model_path, val_loss[j], optimizers[j], schedulers[j])
                
                if val_loss[j] < best_loss[j]:
                    model_path = model_root[j] / "model_joint_best.pt"
                    save(e, networks[j], model_path, val_loss[j], optimizers[j], schedulers[j])
                    best_loss[j] = val_loss[j]

            tq.set_postfix(loss='validation loss={:5f}'.format(torch.mean(val_loss)))
        
        tq.close()
