import sys
import torch
import numpy as np
from pathlib import Path
from dataset import indirectDataset
from network import torqueLstmNetwork, trocarNetwork
from torch.utils.data import DataLoader
from utils import load_model, trocarTester, calculate_force

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1000000
uncorrected_loss = torch.zeros(6)
corrected_loss = torch.zeros(6)
fs_epoch = 1000
trocar_epoch = int(sys.argv[1])
window = 100
skip = 1
path = '../data/csv/test/trocar/with_contact/'
    
#####################################################
## Load free space and trocar model and run
#####################################################

all_pred = None
diff_torque = None
for i in range(6):

    out_joints = [i]
    in_joints = [0,1,2,3,4,5]

    fs_path = "free_space_lstm" + str(i)
    fs_network = torqueLstmNetwork().to(device)
    fs_network = load_model(fs_path, fs_epoch, fs_network, device)

    folder = "trocar_lstm" + str(i)
    network = trocarNetwork(window, len(in_joints), 1).to(device)

    model = trocarTester("trocar", folder, network, window, skip, [i], in_joints, batch_size, device, fs_network, path, is_rnn=True, use_jaw=False)

    print("Loaded a " + str(i) + " model")
    model.load_prev(trocar_epoch)

    uncorrected_loss[i], corrected_loss[i], torque, fs_pred, pred, jacobian, time = model.test(True)
    torque = torque[:,-1,:]
    
    if all_pred is None:
        all_pred = pred
        diff_torque = torque - pred
    else:
        all_pred = torch.cat((all_pred, pred), axis=1)
        diff_torque = torch.cat((diff_torque, torque-pred), axis=1)

#############################################
## Save results and print out
#############################################

# path = Path('../results/with_contact/uncorrected_torques')
# try:
#     path.mkdir(mode=0o777, parents=False)
# except OSError:
#     print("Result path exists")
    
# np.savetxt(path / 'fs_pred_lstm.csv', fs_pred)


# path = Path('../results/with_contact/corrected_torques')
# try:
#     path.mkdir(mode=0o777, parents=False)
# except OSError:
#     print("Result path exists")
    
# np.savetxt(path / 'corrected_pred_lstm.csv', arm_pred)

forces = calculate_force(jacobian, diff_torque)
forces = torch.cat((time.unsqueeze(1), forces), axis=1)
np.savetxt('forces.csv', forces.numpy()) 
        
print('Uncorrected loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (uncorrected_loss[0], uncorrected_loss[1], uncorrected_loss[2], uncorrected_loss[3], uncorrected_loss[4], uncorrected_loss[5], torch.mean(uncorrected_loss)))

print('Corrected loss: t1=%f, t2=%f, f3=%f, t4=%f, t5=%f, t6=%f, mean=%f' % (corrected_loss[0], corrected_loss[1], corrected_loss[2], corrected_loss[3], corrected_loss[4], corrected_loss[5], torch.mean(corrected_loss)))
