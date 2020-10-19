import os
import numpy as np
from os.path import join

def nrmse_loss(y_hat, y):
    denominator = y.max(axis=0)-y.min(axis=0)
    rmse = np.sqrt(np.mean((y_hat-y)**2, axis=0))
    nrmse = rmse/denominator #range_torque[j] #
    return nrmse * 100

test = 'test'
data = 'trocar'
contact = 'no_contact'
results_dir = join('..', 'results', test, data, contact)
joints_dir = join('..', 'data', 'csv', test, data, contact)
joint_names = ['arm', 'insertion', 'platform', 'wrist']
indices = (np.array([0,1]), np.array([0]), np.array([0]), np.array([0,1]))

directories = os.listdir(results_dir)
results = np.zeros((len(directories), 6))

for f in range(len(directories)):
    folder = directories[f]
    cur_joints_path = join(joints_dir, folder, 'joints')
    bag_name = os.listdir(cur_joints_path)
    joints_data = np.loadtxt(join(cur_joints_path,  bag_name[0]), delimiter=',')
    length = joints_data.shape[0]
    pred_torques = np.zeros((length, 6))
    index = 0
    for j in range(len(joint_names)):
        cur_path = join(results_dir, folder, joint_names[j]+'.csv')
        cur_pred = np.loadtxt(cur_path)
        pred_torques[length-cur_pred.shape[0]:, index + indices[j]] = cur_pred[:,indices[j]+1]
        index = index + len(indices[j])

    loss = nrmse_loss(pred_torques, joints_data[:,13:19])
    results[f,:] = loss

print(results)
