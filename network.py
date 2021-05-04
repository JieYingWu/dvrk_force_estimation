import torch
import torch.nn as nn

# Network maps joint position and velocity to torque
class fsNetwork(nn.Module):
    def __init__(self, window, in_joints=6, out_joints=1):
        super(fsNetwork, self).__init__()

        self.layer1 = nn.Linear(window*in_joints*2, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, out_joints)
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.layer4(x)
        x = self.activation(x)
        x = self.layer5(x)
#        x = self.tanh(x)
        return x
    
class trocarNetwork(nn.Module):
    def __init__(self, window, in_joints=6, out_joints=1):
        super(trocarNetwork, self).__init__()

        self.layer1 = nn.Linear(window*in_joints*2+1, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, out_joints)
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
#        x = self.layer2(x)
#        x = self.activation(x)
        x = self.layer3(x)
        x = self.tanh(x)
        return x

# Vaguely inspired by LSTM from https://github.com/BerkeleyAutomation/dvrkCalibration/blob/cec2b8096e3a891c4dcdb09b3161e2a407fee0ee/experiment/3_training/modeling/models.py
class torqueLstmNetwork(nn.Module):
    def __init__(self, batch_size, device, joints=6, hidden_dim=256, num_layers=1):
        super(torqueLstmNetwork, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device
        self.lstm = nn.LSTM(joints*2, hidden_dim, num_layers, batch_first=True)  
        self.linear0 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.linear1 = nn.Linear(int(hidden_dim/2), 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.hidden = self.init_hidden(self.batch_size, self.device)
 
    def forward(self, x):
#        self.hidden = self.init_hidden(x.size()[0], self.device)
#        x, _ = self.lstm(x, self.hidden)
        x, self.hidden = self.lstm(x, self.hidden)
        self.hidden = tuple(state.detach() for state in self.hidden)   
        x = self.linear0(x)
        x = self.relu(x)
        x = self.linear1(x)
#        x = self.tanh(x)
        return x

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers , batch_size, self.hidden_dim).float().to(device),
                torch.zeros(self.num_layers , batch_size, self.hidden_dim).float().to(device))

    
# Network to do direct velocity to force estimate
class forceNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(forceNetwork, self).__init__()

        self.layer1 = nn.Linear(in_channels, 120)
        self.layer2 = nn.Linear(120, 120)
        self.layer3 = nn.Linear(120, 120)
        self.layer4 = nn.Linear(120, 120)
        self.layer5 = nn.Linear(120, out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.layer4(x)
        x = self.activation(x)
        x = self.layer5(x)
        
        return x

