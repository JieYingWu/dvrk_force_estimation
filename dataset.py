from os.path import join
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class indirectDataset(Dataset):
    def __init__(self, path, window, skip, joints = [0,1,2,3,4,5]):

        all_joints = np.array([])
        all_cartesian = np.array([])
        all_jacobian = np.array([])
        joint_path = join(path, 'joints')
        jacobian_path = join(path, 'jacobian')
        for cur_file in os.listdir(joint_path):
#            print(cur_file)
            joints = np.loadtxt(join(joint_path, cur_file), delimiter=',')
            all_joints = np.vstack((all_joints, joints)) if all_joints.size else joints

            jacobian = np.loadtxt(join(jacobian_path, cur_file), delimiter=',')
            all_jacobian = np.vstack((all_jacobian, jacobian)) if all_jacobian.size else jacobian
            
        self.time = all_joints[:,0].astype('int64') # Don't know why the time get written out weird
        self.indices = torch.tensor(joints)
        self.indices = torch.cat((self.indices, self.indices + 7)).tolist()
        self.position_velocity = all_joints[:,self.indices].astype('float32')
        self.torque = all_joints[:,13:19].astype('float32')
        self.jacobian = all_jacobian[:,1:].astype('float32')
        self.window = window
        self.skip = skip
        
    def __len__(self):
        return self.torque.shape[0]/self.window - self.skip

    def __getitem__(self, idx):
        quotient = idx / self.skip
        remainder = idx % self.skip
        begin = quotient * self.window * self.skip + remainder
        end = begin + self.window * self.skip 
        time = self.time[begin:end:self.skip]
        posvel = self.position_velocity[begin:end:self.skip,:].flatten()
        torque = self.torque[end,:]
        jacobian = self.jacobian[end, :]
        return posvel, torque, jacobian

class indirectRnnDataset(Dataset):
    def __init__(self, path, window, skip):

        all_joints = np.array([])
        all_cartesian = np.array([])
        all_jacobian = np.array([])
        joint_path = join(path, 'joints')
#        cartesian_path = join(path, 'cartesian')
        jacobian_path = join(path, 'jacobian')
        for cur_file in os.listdir(joint_path):
            print(cur_file)
            joints = np.loadtxt(join(joint_path, cur_file), delimiter=',')
            all_joints = np.vstack((all_joints, joints)) if all_joints.size else joints
#            cartesian = np.loadtxt(join(cartesian_path, cur_file), delimiter=',')
#            all_cartesian = np.vstack((all_cartesian, cartesian)) if all_cartesian.size else cartesian
            jacobian = np.loadtxt(join(jacobian_path, cur_file), delimiter=',')
            all_jacobian = np.vstack((all_jacobian, jacobian)) if all_jacobian.size else jacobian
            
        self.time = all_joints[:,0].astype('int64') # Don't know why the time get written out weird
        self.position_velocity = all_joints[:,1:13].astype('float32')
        self.torque = all_joints[:,13:19].astype('float32')
#        self.cartesian = all_cartesian[:,1:].astype('float32')
        self.jacobian = all_jacobian[:,1:].astype('float32')
        self.window = window
        self.skip = skip
        
    def __len__(self):
        return self.torque.shape[0]/self.window - self.skip

    def __getitem__(self, idx):
        quotient = idx / self.skip
        remainder = idx % self.skip
        begin = quotient * self.window * self.skip + remainder
        end = begin + self.window * self.skip 
        time = self.time[begin:end:self.skip]
        posvel = self.position_velocity[begin:end:self.skip,:]
        torque = self.torque[end,:]
        #        cartesian = self.cartesian[idx*self.window+self.window,:]
        jacobian = self.jacobian[end, :]
        return posvel, torque, jacobian

    
    
    
class indirectDatasetWithSensor(Dataset):
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
            try:
                sensor = np.loadtxt(join(sensor_path, cur_file), delimiter=',')
            except:
                sensor = None
            if all_sensor is None or sensor is None:
                all_sensor = None
            else:
                all_sensor = np.vstack((all_sensor, sensor)) if all_sensor.size else sensor
            jacobian = np.loadtxt(join(jacobian_path, cur_file), delimiter=',')
            all_jacobian = np.vstack((all_jacobian, jacobian)) if all_jacobian.size else jacobian
            

        self.time = all_joints[:,0].astype('int64') # Don't know why the time get written out weird
        self.position_velocity = all_joints[:,1:13].astype('float32')
        self.torque = all_joints[:,13:19].astype('float32')
        self.sensor = all_sensor[:,1:].astype('float32')
        self.jacobian = all_jacobian[:,1:].astype('float32')
        
    def __len__(self):
        return self.torque.shape[0]

    def __getitem__(self, idx):
        time = self.time[idx]
        posvel = self.position_velocity[idx,:]
        torque = self.torque[idx,:]
        if self.sensor is None:
            sensor = None
        else:
            sensor = self.sensor[idx,:]
        jacobian = self.jacobian[idx, :]
        return posvel, torque, jacobian, sensor

    
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
