from os.path import join
import tqdm
import torch
import torch.nn as nn
from dataset import *
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_torque = (torch.tensor([3.1051168, 2.5269854, 8.118658, 0.06744864, 0.1748129, 0.14781484, 0.08568161])).to(device)
min_torque = (torch.tensor([-3.1137438, -3.1547165, -9.27689, -0.13118862, -0.16299911, -0.12941329,  -0.08511973])).to(device)
range_torque = max_torque - min_torque


def nrmse_loss(y_hat, y, j=0, verbose=False):
    if verbose:
        print(max_torque[j], min_torque[j])
        print(y.max(axis=0).values, y.min(axis=0).values)
        print(y_hat.max(axis=0).values, y_hat.min(axis=0).values)
    denominator = y.max(axis=0).values-y.min(axis=0).values
    rmse = torch.sqrt(torch.mean((y_hat-y)**2, axis=0))
    nrmse = rmse/denominator #range_torque[j] #
    return torch.mean(nrmse) * 100

def relELoss(y_hat, y):
    nominator = torch.sum((y_hat-y)**2)
    demonimator = torch.sum(y**2)
    error = torch.sqrt(nominator/denominator)
    return error * 100

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def load_model(folder, epoch, network, device):
    model_root = Path('checkpoints') / "models_indirect" / folder
    model_path = model_root / 'model_joint_{}.pt'.format(epoch)
    if model_path.exists():
        state = torch.load(str(model_path))
        network.load_state_dict(state['model'])
        network = network.to(device).eval()
    else:
        print('Failed to restore model ' + str(model_path))
        exit()
    return network

def calculate_force(jacobian, joints):
    force = torch.zeros((joints.shape[0], 6))
    for i in range(joints.shape[0]):
        j = jacobian[i].reshape(6,6)
        jacobian_inv_t = torch.from_numpy(np.linalg.inv(j).transpose())
        force[i,:] = torch.matmul(jacobian_inv_t, joints[i])
    return force
        
save = lambda ep, model, model_path, error, optimizer, scheduler: torch.save({
    'model': model.state_dict(),
    'epoch': ep,
    'error': error,
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
}, str(model_path))


class jointTester(object):

    def __init__(self, data, folder, network, window, skip, out_joints,
                 in_joints, batch_size, device, path, is_rnn=False, use_jaw=False):
        dataset = indirectDataset(path, window, skip, in_joints, is_rnn=is_rnn, use_jaw=use_jaw)
        self.loader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle=False)

        self.root = Path('checkpoints' ) 
        self.model_root = self.root / "models_indirect" / folder
        
        self.num_joints = len(out_joints)
        self.joints = out_joints
        self.network = network.to(device)
        self.device = device
        self.loss_fn = nn.MSELoss()#nrmse_loss
        self.is_rnn = is_rnn

    def load_prev(self, epoch):
        if epoch == '0':
            model_path = self.model_root / 'model_joint_best.pt'
        else:
            model_path = self.model_root / 'model_joint_{}.pt'.format(epoch)

        if model_path.exists():

            state = torch.load(str(model_path))
            self.network.load_state_dict(state['model'])
            print('Restored model, epoch {}'.format(epoch))
            self.network.eval()
        else:
            print('Failed to restore model ' + str(model_path))
            exit()
            
    def test(self, verbose=True):
        test_loss = torch.zeros(self.num_joints)
        for i, (position, velocity, torque, jacobian) in enumerate(self.loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            posvel = torch.cat((position, velocity), axis=1).contiguous()
            torque = torque.to(self.device)[:,self.joints]
            
            pred = self.network(posvel).detach() * range_torque[self.joints] + min_torque[self.joints]
            for j in range(self.num_joints):
                loss = self.loss_fn(pred[:,j], torque[:,j],self.joints[j],verbose=True)
                test_loss[j] += loss.item()
                
        if verbose:
            return test_loss, pred.detach().cpu(), jacobian
        return test_loss

class jointLearner(object):
    
    def __init__(self, train_path, val_path, data, folder, network, window, skip, out_joints,
                 in_joints, batch_size, lr, device, is_rnn=False, use_jaw=False):

        self.root = Path('checkpoints' ) 
        self.model_root = self.root / "models_indirect" / folder

        self.num_joints = len(out_joints)
        self.joints = out_joints
        self.network = network.to(device)
        self.batch_size = batch_size
        self.device = device
        self.is_rnn = is_rnn

        try:
            self.model_root.mkdir(mode=0o777, parents=False)
        except OSError:
            print("Model path exists")
        
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, verbose=True)

        train_dataset = indirectDataset(train_path, window, skip, in_joints, is_rnn=is_rnn, use_jaw=use_jaw)
        val_dataset = indirectDataset(val_path, window, skip, in_joints, is_rnn=is_rnn, use_jaw=use_jaw)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size, shuffle=False)

        self.loss_fn = torch.nn.MSELoss()

        init_weights(self.network)
        self.epoch = 1

        self.best_loss = 100

    def load_prev(self, epoch):
        if epoch == '0':
            model_path = self.model_root / 'model_joint_best.pt'
        else:
            model_path = self.model_root / 'model_joint_{}.pt'.format(epoch)
            
        if model_path.exists():
            state = torch.load(str(model_path))
            self.network.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            print('Restored model, epoch {}'.format(epoch))
        else:
            print('Failed to restore model' + str(model_path))
            exit()

        
    def train_step(self, e):
        tq = tqdm.tqdm(total=(len(self.train_loader) * self.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(e, self.optimizer.param_groups[0]['lr']))
        self.network.train()

        epoch_loss = self.step(self.train_loader, tq, train=True)
        
        tq.set_postfix(loss=' loss={:.5f}'.format(epoch_loss/len(self.train_loader)))
        tq.close()

    def val_step(self, e):
        tq = tqdm.tqdm(total=(len(self.val_loader) * self.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(e, self.optimizer.param_groups[0]['lr']))
        self.network.eval()

        val_loss = self.step(self.val_loader, tq, train=False)
        self.scheduler.step(val_loss)
        tq.set_postfix(loss='validation loss={:5f}'.format(val_loss/len(self.val_loader)))

        model_path = self.model_root / "model_joint_{}.pt".format(e)
        save(e, self.network, model_path, val_loss/len(self.val_loader), self.optimizer, self.scheduler)

        if val_loss < self.best_loss:
            model_path = self.model_root / "model_joint_best.pt"
            save(e, self.network, model_path, val_loss/len(self.val_loader), self.optimizer, self.scheduler)
            self.best_loss = val_loss

        tq.close()

    def step(self, loader, tq, train=False):
        step_loss = 0
        for i, (position, velocity, torque, jacobian) in enumerate(loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            posvel = torch.cat((position, velocity), axis=2).contiguous()
            torque = torque.to(self.device)[:, :, self.joints]

            if(self.is_rnn):
                posvel = posvel.permute((1,0,2))
                torque = torque.permute((1,0,2))
                
            pred = self.network(posvel) * range_torque[self.joints] #+ min_torque[self.joints]
            loss = self.loss_fn(pred, torque) 
            step_loss += loss.item()

            if (train):
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            tq.update(self.batch_size)
            tq.set_postfix(loss=' loss={:.5f}'.format(step_loss/(i+1)))

        return step_loss

    
class trocarLearner(jointLearner):
    def __init__(self, train_path, val_path, data, folder, network, window, skip, out_joints,
                 in_joints, batch_size, lr, device, fs_network, is_rnn=False, use_jaw=False):
        super(trocarLearner, self).__init__(train_path, val_path, data, folder, network, window, skip, out_joints, in_joints, batch_size, lr, device, is_rnn=is_rnn, use_jaw=use_jaw)
        
        self.fs_network = fs_network

    def step(self, loader, tq, train=False):
        step_loss = 0
        for i, (position, velocity, torque, jacobian) in enumerate(loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            posvel = torch.cat((position, velocity), axis=2).contiguous()
            torque = torque.to(self.device)[:, :, self.joints]

            if(self.is_rnn):
                fs_posvel = posvel.permute((1,0,2))
            
            fs_torque = self.fs_network(fs_posvel).squeeze().detach() * range_torque[self.joints]

            if(self.is_rnn):
                fs_torque = fs_torque.permute(1,0)
                posvel = posvel.view(posvel.size()[0], -1)

            pred = self.network(posvel) * range_torque[self.joints]
            loss = self.loss_fn(pred.squeeze(), torque[:,-1,0]-fs_torque[:,-1])
            step_loss += loss.item()

            if (train):
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            tq.update(self.batch_size)
            tq.set_postfix(loss=' loss={:.5f}'.format(step_loss/(i+1)))

        return step_loss
    
class trocarTester(jointTester):

    def __init__(self, data, folder, network, window, skip, out_joints,
                 in_joints, batch_size, device, fs_network, path, is_rnn=False, use_jaw=False):
        super(trocarTester, self).__init__(data, folder, network, window, skip, out_joints,
                                           in_joints, batch_size, device, path, is_rnn=is_rnn, use_jaw=use_jaw)
        
        self.fs_network = fs_network

    def test(self, verbose=True):
        uncorrected_loss = torch.zeros(self.num_joints)
        corrected_loss = torch.zeros(self.num_joints)
        for i, (position, velocity, torque, jacobian) in enumerate(self.loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            posvel = torch.cat((position, velocity), axis=2).contiguous()
            torque = torque.to(self.device)[:,:,self.joints]
            
            if(self.is_rnn):
                fs_posvel = posvel.permute((1,0,2))
            fs_torque = self.fs_network(fs_posvel)* range_torque[self.joints]
            
            if(self.is_rnn):
                fs_torque = fs_torque.permute(1,0,2)
                posvel = posvel.view(posvel.size()[0], -1)
                
#            if self.num_joints == 1:
#                fs_torque = fs_torque.unsqueeze(1)

            print(fs_torque.size(), torque.size())

            for j in range(self.num_joints):
                if self.is_rnn:
                    loss = self.loss_fn(fs_torque[:,-1,j], torque[:,-1,j])
                else:
                    loss = self.loss_fn(fs_torque[:,j], torque[:,j])
                uncorrected_loss[j] += loss.item()

            pred = self.network(posvel)* range_torque[self.joints]
            for j in range(self.num_joints):
                if self.is_rnn:
                    loss = self.loss_fn(pred.squeeze(), torque[:,-1,0]-fs_torque[:,-1])
                else:
                    loss = self.loss_fn(pred[:,j]+fs_torque[:,j], torque[:,j])
                corrected_loss[j] += loss.item()

        if verbose:
            return uncorrected_loss, corrected_loss, torque.detach().cpu(), fs_torque.detach().cpu(), pred.detach().cpu(), jacobian
        return uncorrected_loss, corrected_loss
