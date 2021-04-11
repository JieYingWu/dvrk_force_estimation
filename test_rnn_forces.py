import sys
import torch
import numpy as np
from pathlib import Path
from dataset import indirectDataset
from network import torqueLstmNetwork, trocarNetwork
from torch.utils.data import DataLoader
from utils import load_model, trocarTester, calculate_force

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
uncorrected_loss = torch.zeros(6)
corrected_loss = torch.zeros(6)
fs_epoch = 0
trocar_epoch = int(sys.argv[1])
window = 5
skip = 1
exp = sys.argv[2]
path = '../data/csv/test/trocar/with_contact/' + exp
is_rnn = False
    
#####################################################
## Load free space and trocar model and run
#####################################################

all_pred = None
diff_torque = None
for i in range(6):

    out_joints = [i]
    in_joints = [0,1,2,3,4,5]

    fs_path = "lstm/free_space_lstm" + str(i)
    fs_network = torqueLstmNetwork(batch_size, device).to(device)
    fs_network = load_model(fs_path, fs_epoch, fs_network, device)

    folder = "trocar_lstm" + str(i)
    network = trocarNetwork(window, len(in_joints), 1).to(device)

    model = trocarTester(folder, network, window, skip, [i], in_joints, batch_size, device, fs_network, path, is_rnn=is_rnn)

    print("Loaded a " + str(i) + " model")
    model.load_prev(trocar_epoch)

    uncorrected_loss[i], corrected_loss[i], torque, fs_pred, pred, jacobian, time = model.test(True)
    if is_rnn:
        torque = torque[:,-1,:]

    if all_pred is None:
        all_pred = pred
        diff_uncorrected = torque
        diff_corrected = torque - pred
    else:
        all_pred = torch.cat((all_pred, pred), axis=1)
        diff_uncorrected = torch.cat((diff_uncorrected, torque), axis=1)
        diff_corrected = torch.cat((diff_corrected, torque-pred), axis=1)

#############################################
## Save results and print out
#############################################

uncorrected_forces = calculate_force(jacobian, diff_uncorrected)
uncorrected_forces = torch.cat((time.unsqueeze(1), uncorrected_forces), axis=1)
np.savetxt('uncorrected_forces_' + exp + '.csv', uncorrected_forces.numpy()) 

corrected_forces = calculate_force(jacobian, diff_corrected)
corrected_forces = torch.cat((time.unsqueeze(1), corrected_forces), axis=1)
np.savetxt('corrected_forces_' + exp + '.csv', corrected_forces.numpy()) 
        
print('Uncorrected loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (uncorrected_loss[0], uncorrected_loss[1], uncorrected_loss[2], uncorrected_loss[3], uncorrected_loss[4], uncorrected_loss[5], torch.mean(uncorrected_loss)))

print('Corrected loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (corrected_loss[0], corrected_loss[1], corrected_loss[2], corrected_loss[3], corrected_loss[4], corrected_loss[5], torch.mean(corrected_loss)))
