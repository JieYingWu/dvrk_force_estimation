from os.path import join
import numpy as np
import os
import torch
from scipy import signal
from torch.utils.data import Dataset

class indirectDataset(Dataset):
    def __init__(self, path, window, skip, indices = [0,1,2,3,4,5], is_rnn=False, filter_signal=False):

        all_joints = np.array([])
        all_cartesian = np.array([])
        all_jacobian = np.array([])
        joint_path = join(path, 'joints')
        jacobian_path = join(path, 'jacobian')
         
        for cur_file in os.listdir(joint_path):
            joints = np.loadtxt(join(joint_path, cur_file), delimiter=',')
            end_idx = int(joints.shape[0]/window)
            joints = joints[0:end_idx * window, :]
            all_joints = np.vstack((all_joints, joints)) if all_joints.size else joints

            jacobian = np.loadtxt(join(jacobian_path, cur_file), delimiter=',')
            jacobian = jacobian[0:end_idx * window, :]
            all_jacobian = np.vstack((all_jacobian, jacobian)) if all_jacobian.size else jacobian
            
        self.time = all_joints[:,0].astype('float64') # Don't know why the time get written out weird
        self.indices = torch.tensor(indices)
        self.position = all_joints[:,self.indices + 1].astype('float32')
        self.velocity = all_joints[:,self.indices + 7].astype('float32')
        self.torque = all_joints[:,self.indices + 13].astype('float32')

        if len(self.position.shape) < 2:
            self.position = np.expand_dims(self.position, axis=1)
            self.velocity = np.expand_dims(self.velocity, axis=1)
        self.jacobian = all_jacobian[:,1:].astype('float32')
        self.window = window
        self.skip = skip
        self.is_rnn = is_rnn

        ## Filter signals
        if filter_signal:
            b, a = signal.butter(3, 0.02)
            for i in range(len(indices)):
                self.position[:,i] = signal.filtfilt(b, a, self.position[:,i])
                self.velocity[:,i] = signal.filtfilt(b, a, self.velocity[:,i])
                self.torque[:,i] = signal.filtfilt(b, a, self.torque[:,i])
        
    def __len__(self):
        return int(self.torque.shape[0]/self.window) - self.skip

    def __getitem__(self, idx):
        quotient = int(idx / self.skip)
        remainder = idx % self.skip
        begin = quotient * self.window * self.skip + remainder
        end = begin + self.window * self.skip 
        position = self.position[begin:end:self.skip,:]
        velocity = self.velocity[begin:end:self.skip,:]
        if self.is_rnn:
            time = self.time[begin:end:self.skip].flatten()
            torque = self.torque[begin:end:self.skip,:]
        else:
            time = self.time[end-self.skip]
            position = position.flatten()
            velocity = velocity.flatten()
            torque = self.torque[end-self.skip,:]

        #        cartesian = self.cartesian[idx*self.window+self.window,:]
        jacobian = self.jacobian[end-self.skip, :]
        return position, velocity, torque, jacobian, time

class indirectTrocarDataset(indirectDataset):
    def __init__(self, path, window, skip, indices = [0,1,2,3,4,5], is_rnn=False, filter_signal=False):
        super(indirectTrocarDataset, self).__init__(path, window, skip, indices = [0,1,2,3,4,5], is_rnn=is_rnn, filter_signal=filter_signal)
        self.fs_pred = np.loadtxt(path + '/lstm_pred.csv').astype('float32')
        self.fs_pred = self.fs_pred[:,1:]

    def __len__(self):
        return int(self.fs_pred.shape[0]/self.window) - self.skip
        
    def __getitem__(self, idx):
        position, velocity, torque, jacobian, time = super(indirectTrocarDataset, self).__getitem__(idx)
        quotient = int(idx / self.skip)
        remainder = idx % self.skip
        begin = quotient * self.window * self.skip + remainder
        end = begin + self.window * self.skip 
        fs_pred = self.fs_pred[end-self.skip,:]
        return position, velocity, torque, jacobian, time, fs_pred

class indirectTrocarTestDataset(indirectDataset):
    def __init__(self, path, window, skip, indices = [0,1,2,3,4,5], is_rnn=False, filter_signal=False):
        super(indirectTrocarTestDataset, self).__init__(path, window, skip, indices = [0,1,2,3,4,5], is_rnn=is_rnn, filter_signal=filter_signal)
        self.fs_pred = np.loadtxt(path + '/lstm_pred.csv').astype('float32')
        self.fs_pred = self.fs_pred[:,1:]

    def __len__(self):
        return self.fs_pred.shape[0] - self.window*self.skip
        
    def __getitem__(self, idx):
        end = idx + self.window * self.skip 
        position = self.position[idx:end:self.skip,:]
        velocity = self.velocity[idx:end:self.skip,:]
        if self.is_rnn:
            time = self.time[idx:end:self.skip]
            torque = self.torque[idx:end:self.skip,:]
        else:
            time = self.time[end-self.skip]
            position = position.flatten()
            velocity = velocity.flatten()
            torque = self.torque[end-self.skip,:]

        jacobian = self.jacobian[end-self.skip, :]
        fs_pred = self.fs_pred[end-self.skip,:]
        return position, velocity, torque, jacobian, time, fs_pred

