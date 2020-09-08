import tqdm
import torch
import torch.nn as nn
from dataset import indirectDataset
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

def nrmse_loss(y_hat, y):
    print(y.max().item(), y.min().item())
    print(y_hat.max().item(), y_hat.min().item())
    denominator = y.max()-y.min()
    summation = torch.sum((y_hat-y)**2)
    nrmse = torch.sqrt((summation/y.size()[0]))/denominator
    return nrmse * 100

def relELoss(y_hat, y):
    nominator = torch.sum((y_hat-y)**2)
    demonimator = torch.sum(y**2)
    error = torch.sqrt(nominator/denominator)
    return error * 100

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def load_model(root, folder, epoch, network, device):
    model_root = root / "models_indirect" / folder
    model_path = model_root / 'model_joint_{}.pt'.format(epoch)
    if model_path.exists():
        state = torch.load(str(model_path))
        network.load_state_dict(state['model'])
        network = network.to(device).eval()
    else:
        print('Failed to restore model ' + str(model_path))
        exit()
    return network

        
save = lambda ep, model, model_path, error, optimizer, scheduler: torch.save({
    'model': model.state_dict(),
    'epoch': ep,
    'error': error,
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
}, str(model_path))


class jointTester(object):

    def __init__(self, data, folder, network, window, skip, out_joints,
                 in_joints, batch_size, device):
        path = '../data/csv/test/' + data + '/no_contact/'
        dataset = indirectDataset(path, window, skip, in_joints)
        self.loader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle=False)

        self.root = Path('checkpoints' ) 
        self.model_root = self.root / "models_indirect" / folder
        
        self.num_joints = len(out_joints)
        self.joints = out_joints
        self.network = network.to(device)
        self.device = device
        self.loss_fn = nrmse_loss

    def load_prev(self, epoch):
        if epoch == 0:
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
            
    def test(self):
        test_loss = torch.zeros(self.num_joints)
        for i, (position, velocity, torque, jacobian) in enumerate(self.loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            posvel = torch.cat((position, velocity), axis=1)
            torque = torque.to(self.device)[:,self.joints]
            
            pred = self.network(posvel).detach()
            for j in range(self.num_joints):
                loss = self.loss_fn(pred[:,j], torque[:,j]).detach()
                test_loss[j] = loss.item()

        return test_loss

class jointLearner(object):
    
    def __init__(self, data, folder, network, window, skip, out_joints,
                 in_joints, batch_size, lr, device):

        self.train_path = '../data/csv/train/' + data + '/'
        self.val_path = '../data/csv/val/' + data + '/'
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

        train_dataset = indirectDataset(self.train_path, window, skip, in_joints)
        val_dataset = indirectDataset(self.val_path, window, skip, in_joints)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size, shuffle=False)

        self.loss_fn = torch.nn.MSELoss()

        init_weights(self.network)
        self.epoch = 1

        self.best_loss = 100

    def load_prev(self, epoch):
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
        for i, (position, velocity, torque, jacobian) in enumerate(loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            posvel = torch.cat((position, velocity), axis=1)
            torque = torque.to(self.device)[:, self.joints]

            pred = self.network(posvel)
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
    def __init__(self, data, folder, network, window, skip, out_joints,
                 in_joints, batch_size, lr, device, fs_network):
        super(trocarLearner, self).__init__(data, folder, network, window, skip, out_joints,
                 in_joints, batch_size, lr, device)
        
        self.fs_network = fs_network

    def step(self, loader, tq, train=False):
        step_loss = torch.zeros(self.num_joints)
        for i, (position, velocity, torque, jacobian) in enumerate(loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            posvel = torch.cat((position, velocity), axis=1)
            torque = torque.to(self.device)[:, self.joints]

            fs_torque = self.fs_network(posvel).squeeze().detach()

            pred = self.network(posvel)
            loss = self.loss_fn(pred.squeeze(), torque-fs_torque)
            step_loss += loss.item()

            if (train):
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            tq.update(self.batch_size)
            tq.set_postfix(loss=' loss={:.5f}'.format(step_loss))

        return step_loss

class trocarTester(jointTester):

    def __init__(self, data, network, window, skip, out_joints,
                 in_joints, batch_size, device, fs_network):
        super(trocarTester, self).__init__(data, network, window, skip, out_joints,
                                           in_joints, batch_size, device)
        
        self.fs_network = fs_network

    def test(self):
        uncorrected_loss = torch.zeros(self.num_joints)
        corrected_loss = torch.zeros(self.num_joints)
        for i, (position, velocity, torque, jacobian) in enumerate(self.loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            posvel = torch.cat((position, velocity), axis=1)
            torque = torque.to(self.device)[:,self.joints]
            
            fs_torque = self.fs_network(posvel).squeeze()
            for j in range(self.num_joints):
                loss = nrmse_loss(fs_torque[:,j], torque[:,j])
                uncorrected_loss[j] += loss.item()

            pred = self.network(posvel)
            for j in range(self.num_joints):
                loss = self.loss_fn(pred.squeeze()+fs_torque[:,j], torque[:,j])
                corrected_loss[j] += loss.item()

        return uncorrected_loss, corrected_loss
