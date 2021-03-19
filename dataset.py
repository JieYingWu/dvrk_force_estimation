from os.path import join
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class indirectDataset(Dataset):
    def __init__(self, path, window, skip, indices = [0,1,2,3,4,5], is_rnn=False, use_jaw=False):

        all_joints = np.array([])
        all_cartesian = np.array([])
        all_jacobian = np.array([])
        all_jaw = np.array([])
        joint_path = join(path, 'joints')
        jacobian_path = join(path, 'jacobian')
        jaw_path = join(path, 'jaw')
         
        for cur_file in os.listdir(joint_path):
            joints = np.loadtxt(join(joint_path, cur_file), delimiter=',')
            all_joints = np.vstack((all_joints, joints)) if all_joints.size else joints

            if use_jaw:
                jaw = np.loadtxt(join(jaw_path, cur_file), delimiter=',')
                jaw = jaw[0:joints.shape[0],:]
                all_jaw = np.vstack((all_jaw, jaw)) if all_jaw.size else jaw
            
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
        if use_jaw:
            all_jaw = all_jaw.astype('float32')
            self.position = np.concatenate((self.position, all_jaw[:,0:1]), axis=1)
            self.velocity = np.concatenate((self.velocity, all_jaw[:,1:2]), axis=1)
            self.torque = np.concatenate((self.torque, all_jaw[:,2:3]), axis=1)
#        print(np.max(self.torque, axis=0))
#        print(np.min(self.torque, axis=0))
#        exit()
        if len(self.position.shape) < 2:
            self.position = np.expand_dims(self.position, axis=1)
            self.velocity = np.expand_dims(self.velocity, axis=1)
        self.jacobian = all_jacobian[:,1:].astype('float32')
        self.window = window
        self.skip = skip
        self.is_rnn = is_rnn
        
    def __len__(self):
        return int(self.torque.shape[0]/self.window) - self.skip

    def __getitem__(self, idx):
        quotient = int(idx / self.skip)
        remainder = idx % self.skip
        begin = quotient * self.window * self.skip + remainder
        end = begin + self.window * self.skip 
        time = self.time[end-self.skip]
        position = self.position[begin:end:self.skip,:]
        velocity = self.velocity[begin:end:self.skip,:]
        if self.is_rnn:
            torque = self.torque[begin:end:self.skip,:]
        else:
            position = position.flatten()
            velocity = velocity.flatten()
            torque = self.torque[end-self.skip,:]

        #        cartesian = self.cartesian[idx*self.window+self.window,:]
        jacobian = self.jacobian[end-self.skip, :]
        return position, velocity, torque, jacobian, time

class indirectRnnDataset(Dataset):
    def __init__(self, path, window, indices = [0,1,2,3,4,5]):

        self.position = []
        self.velocity = []
        self.torque = []
        self.jacobian = []
        joint_path = join(path, 'joints')
        jacobian_path = join(path, 'jacobian')

        for cur_file in os.listdir(joint_path):
            joints = np.loadtxt(join(joint_path, cur_file), delimiter=',')
            jacobian = np.loadtxt(join(jacobian_path, cur_file), delimiter=',')
            
            for i in range(0,joints.shape[0]-window, window):
                self.position.append(joints[i:i+window,1:7].astype('float32'))
                self.velocity.append(joints[i:i+window,7:13].astype('float32'))
                self.torque.append(joints[i:i+window,13:19].astype('float32'))
                self.jacobian.append(jacobian[i:i+window,1:].astype('float32'))
            
        
    def __len__(self):
        return len(self.position)

    def __getitem__(self, idx):
        position = self.position[idx]
        velocity = self.velocity[idx]
        torque = self.torque[idx]
        jacobian = self.jacobian[idx]
                               
        return position, velocity, torque, jacobian

    
