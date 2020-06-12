import numpy as np
import os
import torch
from torch.utils.data import Dataset

class jointDataset(Dataset):
    def __init__(self, joint_path):

        all_joints = np.array([])
        for cur_file in os.listdir(joint_path):
            cur_path = os.path.join(joint_path, cur_file)
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
