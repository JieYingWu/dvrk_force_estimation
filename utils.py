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
range_torque = (max_torque - min_torque)/2 + 0.2*(max_torque - min_torque)


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
    if epoch == 0:
        model_path = model_root / 'model_joint_best.pt'
    else:
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
        j = jacobian[i,:].reshape(6,6)
        jacobian_inv_t = torch.from_numpy(np.linalg.inv(j).transpose())
        force[i,:] = torch.matmul(jacobian_inv_t, joints[i,:])
    return force
        
save = lambda ep, model, model_path, error, optimizer, scheduler: torch.save({
    'model': model.state_dict(),
    'epoch': ep,
    'error': error,
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
}, str(model_path))


class jointTester(object):

    def __init__(self, folder, network, window, skip, out_joints,
                 in_joints, batch_size, device, path, is_rnn=False, filter_signal=False):

        dataset = indirectDataset(path, window, skip, in_joints, is_rnn=is_rnn, filter_signal=filter_signal)
        self.loader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle=False, drop_last=True)

        self.root = Path('checkpoints' ) 
        self.model_root = self.root / "models_indirect" / folder
        
        self.num_joints = len(out_joints)
        self.joints = out_joints
        self.network = network.to(device)
        self.device = device
        self.loss_fn = nn.MSELoss()#nrmse_loss
        self.is_rnn = is_rnn

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
            print('Failed to restore model ' + str(model_path))
            exit()
            
    def test(self, verbose=True):
        test_loss = torch.zeros(self.num_joints)
        all_torque = None
        all_pred = None
        all_jacobian = None
        all_time = None

        for i, (position, velocity, torque, jacobian, time) in enumerate(self.loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            if self.is_rnn: 
                posvel = torch.cat((position, velocity), axis=2).contiguous()
                torque = torque.to(self.device)[:, :, self.joints]
            else:
                posvel = torch.cat((position, velocity), axis=1).contiguous()
                torque = torque.to(self.device)[:, self.joints]

            if(self.is_rnn):
                posvel = posvel.permute((1,0,2))
                torque = torque.permute((1,0,2))
                time = time.permute((1,0))
                
            pred = self.network(posvel) * range_torque[self.joints] #+ min_torque[self.joints]

            for j in range(self.num_joints):
                loss = self.loss_fn(pred, torque)
                test_loss[j] += loss.item()
                
            if all_torque is None:
                all_torque = torque.detach().cpu()
                all_pred = pred.detach().cpu()
                all_jacobian = jacobian.detach().cpu()
                all_time = time.detach().cpu()
            else:
                all_torque = torch.cat((all_torque, torque.detach().cpu()), axis=0)
                all_pred = torch.cat((all_pred, pred.detach().cpu()), axis=0)
                all_jacobian = torch.cat((all_jacobian, jacobian.detach().cpu()), axis=0)
                all_time = torch.cat((all_time, time.detach().cpu()), axis=0)

        if verbose:
            return test_loss, all_torque, all_pred, jacobian, all_time
        return test_loss

class jointLearner(object):
    
    def __init__(self, train_path, val_path, folder, network, window, skip, out_joints,
                 in_joints, batch_size, lr, device, is_rnn=False, filter_signal=False):

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


#        if is_rnn:
#            train_dataset = indirectRnnDataset(train_path)
#            val_dataset = indirectRnnDataset(val_path)
#        else:
        train_dataset = indirectDataset(train_path, window, skip, in_joints, is_rnn=is_rnn, filter_signal=filter_signal)
        val_dataset = indirectDataset(val_path, window, skip, in_joints, is_rnn=is_rnn, filter_signal=filter_signal)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size, shuffle=False, drop_last=True)

        self.loss_fn = torch.nn.MSELoss()

        init_weights(self.network)
        self.epoch = 1

        self.best_loss = 10000

    def load_prev(self, epoch):
        if epoch == 0:
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
        for i, (position, velocity, torque, jacobian, time) in enumerate(loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            if self.is_rnn: 
                posvel = torch.cat((position, velocity), axis=2).contiguous()
                torque = torque.to(self.device)[:, :, self.joints]
            else:
                posvel = torch.cat((position, velocity), axis=1).contiguous()
                torque = torque.to(self.device)[:, self.joints]

            if(self.is_rnn):
                posvel = posvel.permute((1,0,2))
                torque = torque.permute((1,0,2))
                
            pred = self.network(posvel) * range_torque[self.joints] #+ min_torque[self.joints]
            if self.is_rnn:
                loss = self.loss_fn(pred[100:,:,:], torque[100:,:,:])
            else:
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
    def __init__(self, train_path, val_path, folder, network, window, skip, out_joints,
                 in_joints, batch_size, lr, device, fs_network, is_rnn=False, filter_signal=False):
        super(trocarLearner, self).__init__(train_path, val_path, folder, network, window, skip, out_joints, in_joints, batch_size, lr, device, is_rnn=is_rnn, filter_signal=filter_signal)

        train_dataset = indirectTrocarDataset(train_path, window, skip, in_joints, is_rnn=is_rnn)
        val_dataset = indirectTrocarDataset(val_path, window, skip, in_joints, is_rnn=is_rnn)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size, shuffle=False, drop_last=True)

        
        self.fs_network = fs_network

    def step(self, loader, tq, train=False):
        step_loss = 0
        for i, (position, velocity, torque, jacobian, time, fs_pred) in enumerate(loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            fs_pred = fs_pred.to(self.device)
            if self.is_rnn: 
                posvel = torch.cat((position, velocity, fs_pred), axis=2).contiguous()
                torque = torque.to(self.device)[:, :, self.joints]
                posvel = posvel.view(posvel.size()[0], -1)
            else:
                posvel = torch.cat((position, velocity, fs_pred), axis=1).contiguous()
                torque = torque.to(self.device)[:, self.joints]
                
            pred = self.network(posvel)
            if self.is_rnn:
                loss = self.loss_fn(pred, torque[:,-1,:])
            else:
                loss = self.loss_fn(pred, torque)
            step_loss += loss.item()

            if (train):
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            tq.update(self.batch_size)
            tq.set_postfix(loss=' loss={:.5f}'.format(step_loss/(i+1)))

        return step_loss
    
class trocarTester(jointTester):

    def __init__(self, folder, network, window, skip, out_joints,
                 in_joints, batch_size, device, fs_network, path, is_rnn=False, filter_signal=False):
        super(trocarTester, self).__init__(folder, network, window, skip, out_joints,
                                           in_joints, batch_size, device, path, is_rnn=is_rnn, filter_signal=filter_signal)
        dataset = indirectTrocarDataset(path, window, skip, in_joints, is_rnn=is_rnn, filter_signal=filter_signal)
        self.loader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle=False, drop_last=True)
   
    def test(self, verbose=True):
        uncorrected_loss = torch.zeros(self.num_joints)
        corrected_loss = torch.zeros(self.num_joints)

        all_fs_pred = None
        all_torque = None
        all_pred = None
        all_jacobian = None
        all_time = None
        
        for i, (position, velocity, torque, jacobian, time, fs_pred) in enumerate(self.loader):
            position = position.to(self.device)
            velocity = velocity.to(self.device)
            
            if(self.is_rnn):
                fs_pred = fs_pred.to(self.device)
                posvel = torch.cat((position, velocity, fs_pred), axis=2).contiguous()
                fs_pred = fs_pred[:,:,self.joints]
                posvel = posvel.view(posvel.size()[0], -1)
                torque = torque.to(self.device)[:,:,self.joints]
                time = time[:,-1]
            else:
                fs_pred = fs_pred.to(self.device)
                posvel = torch.cat((position, velocity, fs_pred), axis=1).contiguous()
                fs_pred = fs_pred[:,self.joints]
                torque = torque.to(self.device)[:,self.joints]
                
            pred = self.network(posvel)

            if self.is_rnn:
                pred = pred + fs_pred[:,-1,:]
            else:
                pred = pred + fs_pred                
                
            for j in range(self.num_joints):
                if self.is_rnn:
                    fs_loss = self.loss_fn(fs_pred[:,-1,j], torque[:,-1,j])
                    loss = self.loss_fn(pred[:,-1,j], torque[:,-1,j])

                else:
                    fs_loss = self.loss_fn(fs_pred[:,j], torque[:,j])
                    loss = self.loss_fn(pred[:,j] + fs_pred[:,j], torque[:,j])
                uncorrected_loss[j] += fs_loss.item()
                corrected_loss[j] += loss.item()
            
            uncorrected_loss = uncorrected_loss/len(self.loader)
            corrected_loss = corrected_loss/len(self.loader)
            
            if all_torque is None:
                all_fs_pred = fs_pred.detach().cpu()
                all_torque = torque.detach().cpu()
                all_pred = pred.detach().cpu()
                all_jacobian = jacobian.detach().cpu()
                all_time = time.detach().cpu()
            else:
                all_torque = torch.cat((all_torque, torque.detach().cpu()), axis=0)
                all_fs_pred = torch.cat((all_fs_pred, fs_pred.detach().cpu()), axis=0)
                all_pred = torch.cat((all_pred, pred.detach().cpu()), axis=0)
                all_jacobian = torch.cat((all_jacobian, jacobian.detach().cpu()), axis=0)
                all_time = torch.cat((all_time, time.detach().cpu()), axis=0)

        if verbose:
            return uncorrected_loss, corrected_loss, all_torque, all_fs_pred, all_pred, all_jacobian, all_time
        return uncorrected_loss, corrected_loss
