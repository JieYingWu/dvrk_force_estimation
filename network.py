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

