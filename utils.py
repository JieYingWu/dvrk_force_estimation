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

def load_model(root, folder, epoch, network, j, device):
    model_root = root / "models_indirect" / folder
    model_path = model_root / 'model_joint_{}_{}.pt'.format(j, epoch)
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

    def __init__(self, data, networks, window, skip, out_joints,
                 in_joints, batch_size, device):
        path = '../data/csv/test/' + data + '/no_contact/'
        dataset = indirectDataset(path, window, skip, in_joints)
        self.loader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle=False)

        self.joints = out_joints
        self.num_joints = len(out_joints)
        self.networks = networks
        self.device = device
        self.loss_fn = nrmse_loss

    def test(self):
        test_loss = torch.zeros(self.num_joints)
        for i, (position, velocity, torque, jacobian) in enumerate(self.loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            posvel = torch.cat((position, velocity), axis=1)
            torque = torque.to(self.device)[:,self.joints]
            
            for j in range(self.num_joints):
                pred = self.networks[j](posvel).detach()
                loss = self.loss_fn(pred.squeeze(), torque[:,j]).detach()
                test_loss[j] += loss.item()

        return test_loss

class jointLearner(object):
    
    def __init__(self, data, folder, networks, window, skip, out_joints,
                 in_joints, batch_size, lr, device):

        self.train_path = '../data/csv/train/' + data + '/'
        self.val_path = '../data/csv/val/' + data + '/'
        self.root = Path('checkpoints' ) 
        self.model_root = self.root / "models_indirect" / folder

        self.num_joints = len(out_joints)
        self.joints = out_joints
        self.networks = networks
        self.batch_size = batch_size
        self.device = device

        try:
            self.model_root.mkdir(mode=0o777, parents=False)
        except OSError:
            print("Model path exists")
        
        self.optimizers = []
        self.schedulers = []
        for j in range(self.num_joints):
            self.optimizers.append(torch.optim.SGD(self.networks[j].parameters(), lr))
            self.schedulers.append(ReduceLROnPlateau(self.optimizers[j], verbose=True))

        train_dataset = indirectDataset(self.train_path, window, skip, in_joints)
        val_dataset = indirectDataset(self.val_path, window, skip, in_joints)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size, shuffle=False)

        self.loss_fn = torch.nn.MSELoss()

        for j in range(self.num_joints):
            init_weights(self.networks[j])
        self.epoch = 1


    def load_prev(self, epoch):
        for j in range(self.num_joints):
            model_path = self.model_root / 'model_joint_{}_{}.pt'.format(j, epoch)
            if model_path.exists():
                state = torch.load(str(model_path))
                self.networks[j].load_state_dict(state['model'])
                self.optimizers[j].load_state_dict(state['optimizer'])
                self.schedulers[j].load_state_dict(state['scheduler'])
                print('Restored model, epoch {}'.format(epoch))
            else:
                print('Failed to restore model')
                exit()

        
    def train_step(self, e):
        tq = tqdm.tqdm(total=(len(self.train_loader) * self.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(e, self.optimizers[0].param_groups[0]['lr']))

        for j in range(self.num_joints):
            self.networks[j].train()

        step_loss = self.step(self.train_loader, tq, train=True)
        epoch_loss = torch.sum(step_loss)
        
        tq.set_postfix(loss=' loss={:.5f}'.format(epoch_loss/len(self.train_loader)))
        tq.close()

    def val_step(self, e):
        tq = tqdm.tqdm(total=(len(self.val_loader) * self.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(e, self.optimizers[0].param_groups[0]['lr']))
        for j in range(self.num_joints):
            self.networks[j].eval()

        val_loss = self.step(self.val_loader, tq, train=False)

        for j in range(self.num_joints):
            self.schedulers[j].step(val_loss[j])
        tq.set_postfix(loss='validation loss={:5f}'.format(torch.mean(val_loss)/len(self.val_loader)))

        for j in range(self.num_joints):
            model_path = self.model_root / "model_joint_{}_{}.pt".format(j, e)
            save(e, self.networks[j], model_path, val_loss[j]/len(self.val_loader), self.optimizers[j], self.schedulers[j])

        tq.close()

    def step(self, loader, tq, train=False):
        step_loss = torch.zeros(self.num_joints)
        for i, (position, velocity, torque, jacobian) in enumerate(loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            posvel = torch.cat((position, velocity), axis=1)
            torque = torque.to(self.device)[:, self.joints]

            for j in range(self.num_joints):
                pred = self.networks[j](posvel)
                loss = self.loss_fn(pred.squeeze(), torque[:,j])
                step_loss[j] += loss.item()

                if (train):
                    self.optimizers[j].zero_grad()
                    loss.backward()
                    self.optimizers[j].step()

            tq.update(self.batch_size)
            tq.set_postfix(loss=' loss={:.5f}'.format(torch.sum(step_loss)/(i+1)))

        return step_loss

    
class trocarLearner(jointLearner):
    def __init__(self, data, folder, networks, window, skip, out_joints,
                 in_joints, batch_size, lr, device, fs_networks):
        super(trocarLearner, self).__init__(data, folder, networks, window, skip, out_joints,
                 in_joints, batch_size, lr, device)
        
        self.fs_networks = fs_networks


    def step(self, loader, tq, train=False):
        step_loss = torch.zeros(self.num_joints)
        for i, (position, velocity, torque, jacobian) in enumerate(loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            posvel = torch.cat((position, velocity), axis=1)
            torque = torque.to(self.device)[:, self.joints]

            fs_torque = torch.zeros(posvel.size()[0], self.num_joints).to(self.device)
            for j in range(self.num_joints):
                fs_torque[:,j] = self.fs_networks[j](posvel).squeeze().detach()

            torque = torque - fs_torque
            for j in range(self.num_joints):
                pred = self.networks[j](posvel)
                loss = self.loss_fn(pred.squeeze(), torque[:,j])
                step_loss[j] += loss.item()

                if (train):
                    self.optimizers[j].zero_grad()
                    loss.backward()
                    self.optimizers[j].step()

            tq.update(self.batch_size)
            tq.set_postfix(loss=' loss={:.5f}'.format(torch.sum(step_loss)))

        return step_loss

class trocarTester(jointTester):

    def __init__(self, data, networks, window, skip, out_joints,
                 in_joints, batch_size, device, fs_networks):
        super(trocarTester, self).__init__(data, networks, window, skip, out_joints,
                                           in_joints, batch_size, device)
        
        self.fs_networks = fs_networks

    def test(self):
        test_loss = torch.zeros(self.num_joints)
        for i, (position, velocity, torque, jacobian) in enumerate(self.loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            posvel = torch.cat((position, velocity), axis=1)
            torque = torque.to(self.device)[:,self.joints]
            
            fs_torque = torch.zeros(posvel.size()[0], self.num_joints).to(self.device)
            for j in range(self.num_joints):
                fs_torque[:,j] = self.fs_networks[j](posvel).squeeze()
        
            for j in range(self.num_joints):
                pred = self.networks[j](posvel)
                loss = self.loss_fn(pred.squeeze()+fs_torque[:,j], torque[:,j])
                test_loss[j] += loss.item()

        return test_loss
