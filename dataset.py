from os.path import join
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class jointDataset(Dataset):
    def __init__(self, joint_path):

        all_joints = np.array([])
        for cur_file in os.listdir(joint_path):
            print(cur_file)
            cur_path = join(joint_path, cur_file)
            cur_mat = np.loadtxt(cur_path, delimiter=',')
            all_joints = np.vstack((all_joints, cur_mat)) if all_joints.size else cur_mat
        
        self.position_velocity = all_joints[:,1:13].astype('float32')
        self.torque = all_joints[:,13:19].astype('float32')

    def __len__(self):
        return self.torque.shape[0]

    def __getitem__(self, idx):
        posvel = self.position_velocity[idx,:]
        torque = self.torque[idx,:]
        return posvel, torque

class jointDatasetWithSensor(Dataset):
    def __init__(self, path):

        all_joints = np.array([])
        all_sensor = np.array([])
        all_jacobian = np.array([])
        joint_path = join(path, 'joints')
        sensor_path = join(path, 'sensor')
        jacobian_path = join(path, 'jacobian')
        for cur_file in os.listdir(joint_path):
            print(cur_file)
            joints = np.loadtxt(join(joint_path, cur_file), delimiter=',')
            all_joints = np.vstack((all_joints, joints)) if all_joints.size else joints
#            sensor = np.loadtxt(join(sensor_path, cur_file), delimiter=',')
#            all_sensor = np.vstack((all_sensor, sensor)) if all_sensor.size else sensor
            jacobian = np.loadtxt(join(jacobian_path, cur_file), delimiter=',')
            all_jacobian = np.vstack((all_jacobian, jacobian)) if all_jacobian.size else jacobian
            

        self.time = all_joints[:,0].astype('int64') # Don't know why the time get written out weird
        self.position_velocity = all_joints[:,1:13].astype('float32')
        self.torque = all_joints[:,13:19].astype('float32')
        self.sensor = all_sensor.astype('float32')
        self.jacobian = all_jacobian[:,1:].astype('float32')
        
    def __len__(self):
        return self.torque.shape[0]

    def __getitem__(self, idx):
        time = self.time[idx]
        posvel = self.position_velocity[idx,:]
        torque = self.torque[idx,:]
#        sensor = self.sensor[idx,:]
        jacobian = self.jacobian[idx, :].reshape(6,6)
        jacobian_inv_t = np.linalg.inv(jacobian).transpose()
        return posvel, torque, jacobian_inv_t, time
