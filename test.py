import sys
import torch
from network import *
import torch.nn as nn
import utils
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
contact = 'no_contact'
data = 'free_space'

epoch_to_use = int(sys.argv[1])
exp = sys.argv[2]
net = sys.argv[3]
preprocess = sys.argv[4]
is_rnn = net == 'lstm'
if is_rnn:
    batch_size = 1
else:
    batch_size = 128
    
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
        folder = "free_space" + str(joint)
        
        if is_rnn:
            window = 1000
            network = torqueLstmNetwork(batch_size, device).to(device)
        else:
            window = utils.WINDOW
            network = fsNetwork(window)
        
        model = utils.jointTester(folder, network, window, utils.SKIP, [joint], in_joints, batch_size, device, path, is_rnn=is_rnn, preprocess=preprocess)

        print("Loaded a " + str(joint) + " model")
        model.load_prev(epoch_to_use)

        loss, torque, pred, jacobian, time = model.test(True)
        
        if all_pred is None:
            all_pred = time.unsqueeze(1)

        print(all_pred.size())
        all_pred = torch.cat((all_pred, pred.unsqueeze(1)), axis=1)

    np.savetxt(path + net + '_pred_' + preprocess + '.csv', all_pred.numpy())

        
    print('Loss: ', loss)

if __name__ == "__main__":
    main()
