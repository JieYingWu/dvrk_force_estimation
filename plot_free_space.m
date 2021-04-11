clear
data = 'trocar';
contact = 'no_contact';
test_folder = 'test';
file = 0;
exp = ['exp',num2str(file)];
if strcmp(test_folder, 'test')
    joint_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/joints/'];
    torque_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/lstm_pred.csv'];
else
    joint_path = ['../data/csv/', test_folder, '/', data, '/joints/'];
    torque_path = ['../data/csv/', test_folder, '/', data, '/lstm_pred.csv'];
end
pred_path = 'no_contact_pred.csv';

joint_folder = dir(joint_path);
joint_data = readmatrix([joint_path, joint_folder(3).name]);
torque_data = readmatrix(torque_path);
pred_data = readmatrix(pred_path);

axis_to_plot = [1];
measured_torque = joint_data(:,axis_to_plot+13);
fs_pred_torque = torque_data(:,axis_to_plot+1);
pred_torque = pred_data(:,axis_to_plot+1);

figure
plot(joint_data(:, 1), measured_torque, 'r')
title(data)
hold on
plot(torque_data(:, 1), fs_pred_torque, 'b')
plot(pred_data(:, 1), pred_torque, 'g') 
legend('measured', 'predicted_fs', 'predicted')
title('Torque')
