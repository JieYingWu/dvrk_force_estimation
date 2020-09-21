from os.path import join
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class indirectDataset(Dataset):
    def __init__(self, path, window, skip, indices = [0,1,2,3,4,5], rnn=False):

        all_joints = np.array([])
        all_cartesian = np.array([])
        all_jacobian = np.array([])
        joint_path = join(path, 'joints')
#        cartesian_path = join(path, 'cartesian')
        jacobian_path = join(path, 'jacobian')
        for cur_file in os.listdir(joint_path):
#            print(cur_file)
            joints = np.loadtxt(join(joint_path, cur_file), delimiter=',')
            all_joints = np.vstack((all_joints, joints)) if all_joints.size else joints
#            cartesian = np.loadtxt(join(cartesian_path, cur_file), delimiter=',')
#            all_cartesian = np.vstack((all_cartesian, cartesian)) if all_cartesian.size else cartesian
            jacobian = np.loadtxt(join(jacobian_path, cur_file), delimiter=',')
            all_jacobian = np.vstack((all_jacobian, jacobian)) if all_jacobian.size else jacobian
            
        self.time = all_joints[:,0].astype('float64') # Don't know why the time get written out weird
        self.indices = torch.tensor(indices)
        position_indices = self.indices + 1
        velocity_indices = self.indices + 7
#        torque_indices = self.indices + 13
        self.position = all_joints[:,position_indices].astype('float32')
        self.velocity = all_joints[:,velocity_indices].astype('float32')
        self.torque = all_joints[:,13:19].astype('float32')
        if len(self.position.shape) < 2:
            self.position = np.expand_dims(self.position, axis=1)
            self.velocity = np.expand_dims(self.velocity, axis=1)
#        self.cartesian = all_cartesian[:,1:].astype('float32')
        self.jacobian = all_jacobian[:,1:].astype('float32')
        self.window = window
        self.skip = skip
        self.rnn = rnn # Don't flatten input if RNN
        
    def __len__(self):
        return self.torque.shape[0]/self.window - self.skip

    def __getitem__(self, idx):
        quotient = idx / self.skip
        remainder = idx % self.skip
        begin = quotient * self.window * self.skip + remainder
        end = begin + self.window * self.skip 
        time = self.time[end-self.skip]
        position = self.position[begin:end:self.skip,:]
        velocity = self.velocity[begin:end:self.skip,:]
        if not self.rnn:
            position = position.flatten()
            velocity = velocity.flatten()
        torque = self.torque[end-self.skip,:]
        #        cartesian = self.cartesian[idx*self.window+self.window,:]
        jacobian = self.jacobian[end-self.skip, :]
        return position, velocity, torque, jacobian, time

class indirectForceDataset(indirectDataset):
    def __init__(self, path, window, skip, indices = [0,1,2,3,4,5], rnn=False):
        super(indirectForceDataset, self).__init__(path, window, skip, indices, rnn)
        
    def __len__(self):
        return self.torque.shape[0] - (self.window * self.skip)

    def __getitem__(self, idx):
        begin = idx
        end = begin + self.window * self.skip 
        time = self.time[end-self.skip]
        position = self.position[begin:end:self.skip,:]
        velocity = self.velocity[begin:end:self.skip,:]
        if not self.rnn:
            position = position.flatten()
            velocity = velocity.flatten()
        torque = self.torque[end-self.skip,:]
        jacobian = self.jacobian[end-self.skip, :]
        return position, velocity, torque, jacobian, time

