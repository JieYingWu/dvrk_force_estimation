import sys
import time
import torch
from pathlib import Path
from dataset import indirectDataset
from network import torqueLstmNetwork
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import init_weights, SKIP, save, load_prev, max_torque
from os.path import join

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = 'trocar'
train_path = join('..', 'data', 'csv', 'train', data)
val_path = join('..','data','csv','val', data)
root = Path('checkpoints' )
is_rnn = True
net = 'lstm'
folder = 'lstm/free_space'
j = int(sys.argv[1])

lr = 1e-3
batch_size = 32
epochs = 400
use_previous_model = False
epoch_to_use = 10
in_joints = [0,1,2,3,4,5]
f = True
window = 1000
print('Running for is_rnn value: ', is_rnn)

for num in ['360', '480', '600', '720', '840', '960', '1080']:#'120', '240',
    model = 'filtered_torque_' + num + 's'
    n = int(num)

    network = torqueLstmNetwork(batch_size, device).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr)
    scheduler = ReduceLROnPlateau(optimizer, verbose=False)
                          
    train_dataset = indirectDataset(train_path, window, SKIP, in_joints, num=n, is_rnn=is_rnn, filter_signal=f)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
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
 
    init_weights(network)
    epoch = 1

    print('Training for ' + str(epochs))
    best_loss = torch.zeros(6) + 1e8

    for e in range(epoch, epochs + 1):

        epoch_loss = 0
        network.train()
    
        for i, (position, velocity, torque, jacobian, data_time) in enumerate(train_loader):
            position = position.to(device)
            velocity = velocity.to(device)
            torque = torque.to(device)
        
            step_loss = 0

            posvel = torch.cat((position, velocity), axis=2).contiguous()
            pred = network(posvel)
            loss = loss_fn(pred, torque[:,:,[j]])
            step_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    end = time.time()

#    model_path = model_root / "model_joint_best.pt"
#    save(e, network, model_path, step_loss, optimizer, scheduler)
    
    print('Ran for ', end-start, ' for n = ', n)
