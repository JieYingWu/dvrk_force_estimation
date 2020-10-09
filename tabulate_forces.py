import os
import numpy as np
from os.path import join
from scipy import interpolate
import matplotlib.pyplot as plt

def nrmse_loss(y_hat, y):
    denominator = y.max(axis=0)-y.min(axis=0)
    rmse = np.sqrt(np.mean((y_hat-y)**2, axis=0))
    nrmse = rmse/denominator #range_torque[j] #
    return nrmse * 100

test = 'test'
data = 'free_space'
contact = 'with_contact'
results_dir = join('..', 'results', test, data, contact)
joints_dir = join('..', 'data', 'csv', test, data, contact)
joint_names = ['arm', 'insertion', 'platform', 'wrist']
indices = (np.array([0,1]), np.array([0]), np.array([0]), np.array([0,1]))

directories = os.listdir(results_dir)
results = np.zeros((len(directories), 6))

for f in range(len(directories)):
    folder = directories[f]
    cur_joints_path = join(joints_dir, folder, 'joints')
    cur_sensor_path = join(joints_dir, folder, 'sensor')
    bag_name = os.listdir(cur_joints_path)
    joints_data = np.loadtxt(join(cur_joints_path,  bag_name[0]), delimiter=',')
    sensor_data = np.loadtxt(join(cur_sensor_path,  bag_name[0]), delimiter=',')
    length = joints_data.shape[0]
    pred_torques = np.zeros((length, 6))
    index = 0
    for j in range(len(joint_names)):
        cur_path = join(results_dir, folder, joint_names[j]+'.csv')
        cur_pred = np.loadtxt(cur_path)
        pred_torques[length-cur_pred.shape[0]:, index + indices[j]] = cur_pred[:,indices[j]+1]
        index = index + len(indices[j])

    external_torque = joints_data[:,13:19] - pred_torques
    cur_jacobian_path = join(joints_dir, folder, 'jacobian')
    cur_jacobian_data = np.loadtxt(join(cur_jacobian_path,  bag_name[0]), delimiter=',')
    all_forces = np.zeros((length, 6))
    for i in range(length):
        jacobian = np.array(cur_jacobian_data[i,1:]).reshape((6,6))
        jacobian_inv = np.linalg.inv(jacobian)
        jacobian_inv_t = np.transpose(jacobian_inv)
        all_forces[i,:] = np.matmul(jacobian_inv_t,np.transpose(external_torque[i,:]))
                                   
    interpolator = interpolate.interp1d(joints_data[:,0], all_forces, axis=0, fill_value='extrapolate')
    interpolated_forces = interpolator(sensor_data[:,0])
    interpolated_forces[interpolated_forces[:,2]>0,2] = 0 
    loss = nrmse_loss(interpolated_forces, sensor_data[:,1:])
    results[f,:] = loss

print(results)
plt.plot(sensor_data[:,0], interpolated_forces[:,2], 'b')
plt.plot(sensor_data[:,0], sensor_data[:,3], 'r')
plt.legend(['Predicted', 'Measured'])
plt.show()
