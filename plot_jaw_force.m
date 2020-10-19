data = 'free_space';
exp = 'exp0';
jaw_path = ['../data/csv/test_7dof/', data, '/with_contact/', exp, '/jaw/'];
fsr0_path = ['../data/csv/test_7dof/', data, '/with_contact/', exp, '/fsr0/'];
fsr1_path = ['../data/csv/test_7dof/', data, '/with_contact/', exp, '/fsr1/'];
jaw_folder = dir(jaw_path);
jaw_data = readmatrix([jaw_path, jaw_folder(3).name]); 

joint_path = ['../data/csv/test_7dof/', data, '/with_contact/', exp, '/joints/'];
joint_data = readmatrix([joint_path, jaw_folder(3).name]); 

pred_torque = readmatrix(['../results/test_7dof/', data, '/with_contact/', exp, '/jaw.csv']);
fsr0_folder = dir(fsr0_path);
fsr0_data = readmatrix([fsr0_path, fsr0_folder(3).name]);
fsr0 = 0.009*fsr0_data - 4.7138; % From calibration
fsr1_data = readmatrix([fsr1_path, fsr0_folder(3).name]);
fsr1 = 0.0083*fsr1_data - 4.4605; % From calibration

length = size(jaw_data);
length = length(1);
padded_torque = zeros(length, 1);
padded_torque(length - size(pred_torque, 1) + 1:end, 1) = pred_torque(:,2);
external_torque = jaw_data(:,3) - padded_torque;

figure
plot(joint_data(:,1), external_torque, 'r')
hold on
plot(fsr0_data(:,1), fsr0(:,2), 'b')
legend('pred', 'measured')