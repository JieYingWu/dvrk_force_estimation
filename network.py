import torch
import torch.nn as nn

# Network maps joint position and velocity to torque
class fsNetwork(nn.Module):
    def __init__(self, window, in_joints=6, out_joints=1):
        super(fsNetwork, self).__init__()

        self.layer1 = nn.Linear(window*in_joints*2, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, out_joints)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.bn3(x)
        x = self.layer4(x)
        return x
    
class trocarNetwork(nn.Module):
    def __init__(self, window, in_joints=6, out_joints=1):
        super(trocarNetwork, self).__init__()

        self.layer1 = nn.Linear(window*in_joints*2, 512)
        self.layer2 = nn.Linear(512, out_joints)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(64)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.tanh(x)
        return x

# Network maps joint position and velocity to torque
class wristNetwork(nn.Module):
    def __init__(self, window, in_joints=6, out_joints=1):
        super(wristNetwork, self).__init__()

        self.layer1 = nn.Linear(window*in_joints*2, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(256, out_joints)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.bn3(x)
        x = self.layer4(x)
        x = self.activation(x)
        x = self.bn4(x)
        x = self.layer5(x)
#        x = self.sigmoid()
        return x

class wristTrocarNetwork(nn.Module):
    def __init__(self, window, joints=6):
        super(wristTrocarNetwork, self).__init__()

        self.layer1 = nn.Linear(window*joints*2, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 3)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.layer3(x)
        return x

# Network maps joint position and velocity to torque
class insertionNetwork(nn.Module):
    def __init__(self, window, in_joints=6, out_joints=1):
        super(insertionNetwork, self).__init__()

        self.layer1 = nn.Linear(window*in_joints*2, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, out_joints)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.bn3(x)
        x = self.layer4(x)
        return x

class insertionTrocarNetwork(nn.Module):
    def __init__(self, window, joints=6):
        super(insertionTrocarNetwork, self).__init__()

        self.layer1 = nn.Linear(window*joints*2, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 1)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.layer3(x)
        return x

class armNetwork(nn.Module):
    def __init__(self, window, in_joints=6, out_joints=1):
        super(armNetwork, self).__init__()

        self.layer1 = nn.Linear(window*in_joints*2, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(256, out_joints)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.bn3(x)
        x = self.layer4(x)
        x = self.activation(x)
        x = self.bn4(x)
        x = self.layer5(x)
        return x
    
class armTrocarNetwork(nn.Module):
    def __init__(self, window):
        super(armTrocarNetwork, self).__init__()

        self.layer1 = nn.Linear(window*12, 512)
        self.layer2 = nn.Linear(512,256)
        self.layer3 = nn.Linear(256,2)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.layer3(x)
        return x
    
    
# Vaguely inspired by LSTM from https://github.com/BerkeleyAutomation/dvrkCalibration/blob/cec2b8096e3a891c4dcdb09b3161e2a407fee0ee/experiment/3_training/modeling/models.py
class torqueLstmNetwork(nn.Module):
    def __init__(self, joints=6, hidden_dim=128, num_layers=1, p=0.5):
        super(torqueLstmNetwork, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(joints*2, hidden_dim, num_layers)  
        self.linear0 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(int(hidden_dim/2), 1)
        #self.bn0 = nn.BatchNorm1d(int(hidden_dim/2))

    def forward(self, x):
        hidden = self.init_hidden(x)
        x, _ = self.lstm(x, hidden)
        x = self.linear0(x)
        x = self.relu(x)
        #x = self.bn0(x)
        x = self.linear1(x)
        x = self.tanh(x)
        return x

    def init_hidden(self, x):
        if next(self.parameters()).is_cuda:
            return (torch.zeros(self.num_layers , x.size(1), self.hidden_dim).float().cuda(),
                    torch.zeros(self.num_layers , x.size(1), self.hidden_dim).float().cuda())
        return (torch.zeros(self.num_layers , x.size(1), self.hidden_dim).float(),
                torch.zeros(self.num_layers , x.size(1), self.hidden_dim).float())

    
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

