import sys
import torch
from network import *
import torch.nn as nn
import utils
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
contact = 'no_contact'
data = 'trocar'

epoch_to_use = 0 #int(sys.argv[1])
exp = sys.argv[1] #sys.argv[2]
net = 'ff' #sys.argv[3]
preprocess = 'filtered_torque'# sys.argv[4]
filter_signal = False
is_rnn = net == 'lstm'
if is_rnn:
    batch_size = 1
else:
    batch_size = 8192
    
def main():
    all_pred = None
    if exp == 'train':
        path = '../data/csv/train/' + data + '/'
    elif exp == 'val':
        path = '../data/csv/val/' + data + '/'
    elif exp =='test':
        path = '../data/csv/test/' + data + '/' + contact + '/'
    else:
        path = '../data/csv/test/' + data + '/' + contact + '/' + exp + '/'
    in_joints = [0,1,2,3,4,5]
    
    for joint in range(6):
        folder = "no_cannula" + str(joint)
        
        if is_rnn:
            window = 1000
            network = torqueLstmNetwork(batch_size, device).to(device)
        else:
            window = utils.WINDOW
            network = fsNetwork(window)
        
        model = utils.jointTester(folder, network, window, utils.SKIP, [joint], in_joints, batch_size, device, path, is_rnn=is_rnn, filter_signal=filter_signal, preprocess=preprocess)

        print("Loaded a " + str(joint) + " model")
        model.load_prev(epoch_to_use)

        loss, torque, pred, jacobian, time = model.test(True)
        
        if all_pred is None:
            all_pred = time.unsqueeze(1)

        print(all_pred.size())
        all_pred = torch.cat((all_pred, pred.unsqueeze(1)), axis=1)

    np.savetxt(path + net + '_base_pred_' + preprocess + '.csv', all_pred.numpy())

        
    print('Loss: ', loss)

if __name__ == "__main__":
    main()
