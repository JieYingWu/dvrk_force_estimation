import torch
import torch.nn as nn


# Network maps joint position and velocity to torque
class torqueNetwork(nn.Module):
    def __init__(self):
        super(torqueNetwork, self).__init__()

        self.layer1 = nn.Linear(12, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 1)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        
        return x

class torqueWindowNetwork(nn.Module):
    def __init__(self, window):
        super(torqueWindowNetwork, self).__init__()

        self.layer1 = nn.Linear(12*window, 12*window)
        self.layer2 = nn.Linear(12*window, 6*window)
        self.layer3 = nn.Linear(6*window, 3*window)
        self.layer4 = nn.Linear(3*window, window)
        self.layer5 = nn.Linear(window, 1)
        self.activation = nn.ReLU()
        
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


# Vaguely inspired by LSTM from https://github.com/BerkeleyAutomation/dvrkCalibration/blob/cec2b8096e3a891c4dcdb09b3161e2a407fee0ee/experiment/3_training/modeling/models.py
class torqueLstmNetwork(nn.Module):
    def __init__(self, in_channels=12, hidden_dim=100):
        super(torqueLstmNetwork, self).__init__()
        
        self.lstm = nn.LSTM(in_channels, hidden_dim)  
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[-1])
        return out
        
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
#        x = self.activation(x)
#        x = self.layer6(x)
#        x = self.activation(x)
#        x = self.layer7(x)
        
        return x
