data = 'free_space';
jaw_path = ['../data/csv/test_jaw/', data, '/no_contact/jaw/'];
jaw_folder = dir(jaw_path);
jaw_data = readmatrix([jaw_path, jaw_folder(3).name]); 

pred_torque = readmatrix(['../results/no_contact/', data, '_torques/jaw.csv']);
pred_torque = pred_torque(:,2);
measured_torques = jaw_data(:,4);
figure
plot(pred_torque, 'r')
hold on
plot(measured_torques, 'b')
legend('pred', 'measured')