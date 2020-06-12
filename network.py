import torch
import torch.nn as nn


# Network maps joint position and velocity to torque
class torqueNetwork(nn.Module):
    def __init__(self):
        super(torqueNetwork, self).__init__()

        self.layer1 = nn.Linear(12, 100, bias=True)
        self.layer2 = nn.Linear(100, 1, bias=True)

    def forward(self, x):
        layer1 = self.layer1(x)
        layer1 = torch.sigmoid(layer1)
        layer2 = self.layer2(layer1)

        output = layer2

        return output

