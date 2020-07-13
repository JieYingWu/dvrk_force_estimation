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

# Network to do direct velocity to force estimate
class forceNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(forceNetwork, self).__init__()

        self.layer1 = nn.Linear(in_channels, 80)
        self.layer2 = nn.Linear(80, 80)
        self.layer3 = nn.Linear(80, 80)
        self.layer4 = nn.Linear(80, 80)
        self.layer5 = nn.Linear(80, 80)
        self.layer6 = nn.Linear(80, 80)
        self.layer7 = nn.Linear(80, out_channels)
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
        x = self.activation(x)
        x = self.layer6(x)
        x = self.activation(x)
        x = self.layer7(x)
        
        return x
