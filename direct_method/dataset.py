from os.path import join
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class directDataset(Dataset):
    def __init__(self, path):

        all_joints = np.array([])
        all_sensor = np.array([])
        all_cartesian = np.array([])
        joint_path = join(path, 'joints')
        sensor_path = join(path, 'sensor')
        for cur_file in os.listdir(joint_path):
            print(cur_file)
            joints = np.loadtxt(join(joint_path, cur_file), delimiter=',')
            all_joints = np.vstack((all_joints, joints)) if all_joints.size else joints
            sensor = np.loadtxt(join(sensor_path, cur_file), delimiter=',')
            all_sensor = np.vstack((all_sensor, sensor)) if all_sensor.size else sensor

        self.veltorque = all_joints[:,7:19].astype('float32')
        self.sensor = all_sensor[:,1:].astype('float32')
        
    def __len__(self):
        return self.veltorque.shape[0]/10

    def __getitem__(self, idx):
        veltorque = self.veltorque[idx*10:idx*10+10,:].flatten()
        sensor = self.sensor[idx*10+10,:].flatten()
        return veltorque, sensor

    
class directDatasetWithCartesian(Dataset):
    def __init__(self, path):

        all_joints = np.array([])
        all_sensor = np.array([])
        all_cartesian = np.array([])
        joint_path = join(path, 'joints')
        sensor_path = join(path, 'sensor')
        cartesian_path = join(path, 'cartesian')
        for cur_file in os.listdir(joint_path):
            print(cur_file)
            joints = np.loadtxt(join(joint_path, cur_file), delimiter=',')
            all_joints = np.vstack((all_joints, joints)) if all_joints.size else joints
            sensor = np.loadtxt(join(sensor_path, cur_file), delimiter=',')
            all_sensor = np.vstack((all_sensor, sensor)) if all_sensor.size else sensor
            cartesian = np.loadtxt(join(cartesian_path, cur_file), delimiter=',')
            all_cartesian = np.vstack((all_cartesian, cartesian)) if all_cartesian.size else cartesian

        self.veltorque = all_joints[:,7:19].astype('float32')
        self.sensor = all_sensor[:,1:].astype('float32')
        self.cartesian = all_cartesian[:,1:].astype('float32')
        
    def __len__(self):
        return self.veltorque.shape[0]/10

    def __getitem__(self, idx):
        veltorque = self.veltorque[idx*10:idx*10+10,:].flatten()
        sensor = self.sensor[idx*10+10,:].flatten()
        cartesian = self.cartesian[idx*10+10,:]
        return veltorque, sensor, cartesian
