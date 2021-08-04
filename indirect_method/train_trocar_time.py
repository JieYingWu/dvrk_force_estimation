import sys
import time
import torch
from pathlib import Path
from dataset import indirectTrocarDataset
from network import trocarNetwork
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import init_weights, WINDOW, SKIP, save, load_prev, max_torque
from os.path import join

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = 'trocar'
train_path = join('..', 'data', 'csv', 'train', data)
val_path = join('..','data','csv','val', data)
root = Path('checkpoints' )
is_rnn = True
net = 'lstm'
folder = 'lstm/trocar'
j = int(sys.argv[1])

lr = 1e-3
batch_size = 4096
epochs = 400
use_previous_model = False
epoch_to_use = 10
in_joints = [0,1,2,3,4,5]
f = False
print('Running for is_rnn value: ', is_rnn)

for num in ['120', '240', '360', '480', '600', '720', '840', '960', '1080']:
    model = 'filtered_torque_' + num + 's'
    n = int(num)

    network = trocarNetwork(WINDOW, len(in_joints), 1).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr)
    scheduler = ReduceLROnPlateau(optimizer, verbose=False)

    train_dataset = indirectTrocarDataset(train_path, WINDOW, SKIP, in_joints, num=n, filter_signal=f, net=net)
    val_dataset = indirectTrocarDataset(val_path, WINDOW, SKIP, in_joints, num=n, filter_signal=f, net=net)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size, shuffle=False)
    
    loss_fn = torch.nn.MSELoss()

    start = time.time()

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

    try:
        model_root = root / model / (folder + str(j))
        model_root.mkdir(mode=0o777, parents=False)
    except OSError:
        print("Model path exists")
 
    # Read existing weights for both G and D models
    init_weights(network)
    epoch = 1

    print('Training for ' + str(epochs))
    best_loss = torch.zeros(6) + 1e8

    for e in range(epoch, epochs + 1):

        epoch_loss = 0
        network.train()
    
        for i, (position, velocity, torque, data_time, fs_pred) in enumerate(train_loader):
            position = position.to(device)
            velocity = velocity.to(device)
            fs_pred = fs_pred.to(device)
            torque = torque.to(device)
        
            step_loss = 0

            posvel = torch.cat((position, velocity, fs_pred[:,[j]]), axis=1).contiguous()
            pred = network(posvel) + fs_pred[:,[j]]
            loss = loss_fn(pred, torque[:,[j]])
            step_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    end = time.time()

    #model_path = model_root / "model_joint_best.pt"
    #save(e, network, model_path, step_loss, optimizer, scheduler)

    print('Ran for ', end-start)
