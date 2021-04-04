data = 'trocar';
contact = 'with_contact';
test_folder = 'test';
file = 1;
exp = ['exp',num2str(file)];
joint_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/joints/'];
force_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/sensor/'];

joint_folder = dir(joint_path);
force_data = readmatrix([force_path, joint_folder(3).name]);

%data = 'corrected';
uncorrected_pred_forces = readmatrix(['uncorrected_forces_', exp, '.csv']);
corrected_pred_forces = readmatrix(['corrected_forces_', exp, '.csv']);

axis_to_plot = [2];
uncorrected_predicted = uncorrected_pred_forces(:,axis_to_plot+1);
corrected_predicted = corrected_pred_forces(:,axis_to_plot+1);
%predicted(predicted > 0) = 0;

figure
plot(uncorrected_pred_forces(:, 1), uncorrected_predicted, 'r')
title(data)
hold on
plot(corrected_pred_forces(:, 1), corrected_predicted, 'g')
plot(force_data(:, 1), force_data(:,axis_to_plot+1), 'b')
legend('uncorrected', 'corrected', 'measured')
title('Force')

