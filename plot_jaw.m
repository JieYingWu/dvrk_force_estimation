data = 'free_space';
exp = 'exp3';
jaw_path = ['../data/csv/test/', data, '/no_contact/', exp, '/jaw/'];
jaw_folder = dir(jaw_path);
jaw_data = readmatrix([jaw_path, jaw_folder(3).name]); 

pred_torque = readmatrix(['../results/test/', data, '/no_contact/', exp, '/jaw.csv']);
pred_torque = pred_torque(:,2);
measured_torques = jaw_data(:,3);
figure
plot(pred_torque, 'r')
hold on
plot(measured_torques, 'b')
legend('pred', 'measured')