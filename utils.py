from os.path import join
import tqdm
import torch
import torch.nn as nn
from dataset import *
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_torque = (torch.tensor([ 3.1051168, 2.704937, 10.227721, 0.06926354, 0.17762128, 0.15118882])).to(device)
min_torque = (torch.tensor([-3.1137438, -3.1953104, -9.27689, -0.13118862, -0.17046826, -0.13808922])).to(device)
range_torque = max_torque - min_torque

def nrmse_loss(y_hat, y, j, verbose=False):
    if verbose:
        print(max_torque[j], min_torque[j])
        print(y_hat.max(axis=0).values, y_hat.min(axis=0).values)
#    denominator = y.max(axis=0).values-y.min(axis=0).values
    summation = torch.sum((y_hat-y)**2, axis=0)
    nrmse = torch.sqrt((summation/y.size()[0]))/range_torque[j] #denominator #
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
                 in_joints, batch_size, device, path):
        dataset = indirectForceDataset(path, window, skip, in_joints)
        self.loader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle=False)

        self.root = Path('checkpoints' ) 
        self.model_root = self.root / "models_indirect" / folder
        
        self.num_joints = len(out_joints)
        self.joints = out_joints
        self.network = network.to(device)
        self.device = device
        self.loss_fn = nrmse_loss

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
            print('Failed to restore model')
            exit()
            
    def test(self, verbose=True):
        test_loss = torch.zeros(self.num_joints)
        for i, (position, velocity, torque, jacobian, time) in enumerate(self.loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            posvel = torch.cat((position, velocity), axis=1).contiguous()
            torque = torque.to(self.device)[:,self.joints]
            
            pred = self.network(posvel).detach() 
            for j in range(self.num_joints):
                loss = self.loss_fn(pred[:,j], torque[:,j],self.joints[j],verbose=True)
                test_loss[j] += loss.item()
                
        if verbose:
            return test_loss, pred.detach().cpu(), jacobian, time
        return test_loss

class jointLearner(object):
    
    def __init__(self, train_path, val_path, data, folder, network, window, skip, out_joints,
                 in_joints, batch_size, lr, device):

        self.root = Path('checkpoints' ) 
        self.model_root = self.root / "models_indirect" / folder

        self.num_joints = len(out_joints)
        self.joints = out_joints
        self.network = network.to(device)
        self.batch_size = batch_size
        self.device = device

        try:
            self.model_root.mkdir(mode=0o777, parents=False)
        except OSError:
            print("Model path exists")
        
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, verbose=True)

        train_dataset = indirectDataset(train_path, window, skip, in_joints)
        val_dataset = indirectDataset(val_path, window, skip, in_joints)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size, shuffle=False)

        self.loss_fn = nrmse_loss

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
            print('Failed to restore model')
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
        for i, (position, velocity, torque, jacobian, time) in enumerate(loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            posvel = torch.cat((position, velocity), axis=1).contiguous()
            torque = torque.to(self.device)[:, self.joints]

            pred = self.network(posvel)
            loss = self.loss_fn(pred, torque, self.joints) 
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
                 in_joints, batch_size, lr, device, fs_network):
        super(trocarLearner, self).__init__(train_path, val_path, data, folder, network, window, skip, out_joints, in_joints, batch_size, lr, device)
        
        self.fs_network = fs_network

    def step(self, loader, tq, train=False):
        step_loss = 0
        for i, (position, velocity, torque, jacobian, time) in enumerate(loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            posvel = torch.cat((position, velocity), axis=1).contiguous()
            torque = torque.to(self.device)[:, self.joints].squeeze()

            fs_torque = self.fs_network(posvel).squeeze().detach()

            pred = self.network(posvel)
            loss = self.loss_fn(pred.squeeze(), torque-fs_torque)
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
                 in_joints, batch_size, device, fs_network, path):
        super(trocarTester, self).__init__(data, folder, network, window, skip, out_joints,
                                           in_joints, batch_size, device, path)
        
        self.fs_network = fs_network

    def test(self, verbose=True):
        uncorrected_loss = torch.zeros(self.num_joints)
        corrected_loss = torch.zeros(self.num_joints)
        for i, (position, velocity, torque, jacobian, time) in enumerate(self.loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            posvel = torch.cat((position, velocity), axis=1).contiguous()
            torque = torque.to(self.device)[:,self.joints]
            
            fs_torque = self.fs_network(posvel).squeeze()
            if self.num_joints == 1:
                fs_torque = fs_torque.unsqueeze(1)
            for j in range(self.num_joints):
                loss = loss_fn(fs_torque[:,j], torque[:,j])
                uncorrected_loss[j] += loss.item()

            pred = self.network(posvel)
            for j in range(self.num_joints):
                loss = self.loss_fn(pred[:,j]+fs_torque[:,j], torque[:,j])
                corrected_loss[j] += loss.item()

        if verbose:
            return uncorrected_loss, corrected_loss, torque.detach().cpu(), fs_torque.detach().cpu(), pred.detach().cpu(), jacobian, time
        return uncorrected_loss, corrected_loss
    


class jawLearner(jointLearner):
    
    def __init__(self, train_path, val_path, data, folder, network, window, skip, out_joints,
                 in_joints, batch_size, lr, device):

        super(jawLearner, self).__init__(train_path, val_path, data, folder, network, window, skip, out_joints,
                                           in_joints, batch_size, lr, device)
        self.train_path = join('..', 'data', 'csv', 'train_jaw', data)
        self.val_path = join('..','data','csv','val_jaw', data)

        train_dataset = indirectJawDataset(self.train_path, window, skip, in_joints)
        val_dataset = indirectJawDataset(self.val_path, window, skip, in_joints)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size, shuffle=False)

class jawTester(jointTester):

    def __init__(self, data, folder, network, window, skip, out_joints,
                 in_joints, batch_size, device, path):
        super(jawTester, self).__init__(data, folder, network, window, skip, out_joints,
                 in_joints, batch_size, device, path)
        dataset = indirectJawForceDataset(path, window, skip, in_joints)
        self.loader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle=False)
