import os
import sys
from os.path import join
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

all_joints = np.array([])
all_jacobian = np.array([])

data = sys.argv[1]
split = sys.argv[2]
contact = 'no_contact'
exp = sys.argv[2]
path = join('..', 'data', 'csv', split, data)#, contact, exp)

joint_path = join(path, 'joints')
jacobian_path = join(path, 'jacobian')
cut_off = 1000

for cur_file in os.listdir(joint_path):
    joints = np.loadtxt(join(joint_path, cur_file), delimiter=',')
    end_idx = int(joints.shape[0]/cut_off)
    joints = joints[0:end_idx * cut_off, :]
    if joints[0,0] > 10000:
        joints[:,0] = joints[:,0] - joints[0,0]
    if all_joints.size:
        joints[:,0] = all_joints[-1,0] + joints[:,0] + 0.0001
    all_joints = np.vstack((all_joints, joints)) if all_joints.size else joints

    jacobian = np.loadtxt(join(jacobian_path, cur_file), delimiter=',')
    jacobian = jacobian[0:end_idx * cut_off, :]
    if jacobian[0,0] > 10000:
        jacobian[:,0] = jacobian[:,0] - jacobian[0,0]
    if all_jacobian.size:
        jacobian[:,0] = all_jacobian[-1,0] + jacobian[:,0] + 0.0001
    all_jacobian = np.vstack((all_jacobian, jacobian)) if all_jacobian.size else jacobian

    
print(all_joints.shape, all_jacobian.shape)
joint_time = all_joints[:,0] - all_joints[0,0]
jacobian_time = all_jacobian[:,0] - all_joints[0,0]

start_time = np.max([joint_time[0], jacobian_time[0]])
end_time = np.min([joint_time[-1], jacobian_time[-1]])
print(start_time, end_time)

interpolated_time = np.arange(start_time, end_time, 0.001)
interp_joints = np.zeros((interpolated_time.shape[0], 19))
interp_joints[:,0] = interpolated_time

for i in range(18):
    f = interpolate.interp1d(joint_time, all_joints[:,i+1])
    interp_joints[:,i+1] = f(interpolated_time)

np.savetxt(join(joint_path,'interpolated_all_joints.csv'), interp_joints, delimiter=',')

interp_jacobian = np.zeros((interpolated_time.shape [0], 37))
interp_jacobian[:,0] = interpolated_time
for i in range(36):
    f = interpolate.interp1d(jacobian_time, all_jacobian[:,i+1])
    interp_jacobian[:,i+1] = f(interpolated_time)

np.savetxt(join(jacobian_path,'interpolated_all_jacobian.csv'), interp_jacobian, delimiter=',')
