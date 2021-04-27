clear
data = 'trocar';
contact = 'no_contact';
test_folder = 'test';
file = 0;
seal = 'seal';
exp = ['exp',num2str(file)];
if strcmp(test_folder, 'test')
    joint_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/joints/'];
    fs_pred_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/ff_', seal, '_pred_filtered_torque.csv'];
else
    joint_path = ['../data/csv/', test_folder, '/', data, '/joints/'];
    fs_pred_path = ['../data/csv/', test_folder, '/', data, '/ff_', seal, '_pred_filtered_torque.csv'];
end
pred_path = ['../results/', data, '/', contact, '/', 'torque_', seal, '_pred.csv'];

joint_folder = dir(joint_path);
joint_data = readmatrix([joint_path, 'interpolated_all_joints.csv']);
fs_pred_data = readmatrix(fs_pred_path);
pred_data = readmatrix(pred_path);

axis_to_plot = [6];
measured_torque = joint_data(:,axis_to_plot+13);
fs_pred_torque = fs_pred_data(:,axis_to_plot+1);
pred_torque = pred_data(:,axis_to_plot+1);
fs_loss = sqrt(mean((measured_torque(1:length(fs_pred_torque)) - fs_pred_torque).^2))
loss = sqrt(mean((measured_torque(1:length(pred_torque)) - pred_torque).^2))

figure
plot(fs_pred_data(:, 1), fs_pred_torque, 'r')
title(data)
hold on
plot(pred_data(:, 1), pred_torque, 'b') 
plot(joint_data(:, 1), measured_torque, 'k')
legend('predicted_fs', 'predicted', 'measured')
title('Torque')
