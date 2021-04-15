clear
data = 'trocar';
contact = 'with_contact';
test_folder = 'test';
file = 1;
exp = ['exp',num2str(file)];
joint_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/joints/'];
force_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/sensor/'];

joint_folder = dir(joint_path);
joint_data = readmatrix([joint_path, joint_folder(3).name]);
force_data = readmatrix([force_path, joint_folder(3).name]);

uncorrected_pred_forces = readmatrix(['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/', 'uncorrected_forces.csv']);
corrected_pred_forces = readmatrix(['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/', 'corrected_forces.csv']);
pred_torque = readmatrix(['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/', 'pred.csv']);

axis_to_plot = [1];
uncorrected_predicted = uncorrected_pred_forces(:,axis_to_plot+1);
corrected_predicted = corrected_pred_forces(:,axis_to_plot+1);

figure
plot(uncorrected_pred_forces(:, 1), uncorrected_predicted, 'r')
title(data)
hold on
plot(corrected_pred_forces(:, 1), corrected_predicted, 'g') 
plot(force_data(:, 1), force_data(:,axis_to_plot+1), 'b')
legend('uncorrected', 'corrected', 'measured')
title('Force')

figure
plot(pred_torque(:,axis_to_plot), 'r')
hold on
plot(joint_data(:,axis_to_plot+14), 'b')
