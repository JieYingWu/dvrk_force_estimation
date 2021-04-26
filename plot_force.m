clear
data = 'trocar';
contact = 'with_contact';
test_folder = 'test';
net = 'ff';
preprocess = 'filtered_torque_30s';
file = 3;
exp = ['exp',num2str(file)];
joint_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/joints/'];
force_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/sensor/'];

joint_folder = dir(joint_path);
joint_data = readmatrix([joint_path, 'interpolated_all_joints']);
force_data = readmatrix([force_path, joint_folder(3).name]);

uncorrected_pred_forces = readmatrix(['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/', 'uncorrected_forces_', net, '_', preprocess, '.csv']);
corrected_pred_forces = readmatrix(['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/', 'corrected_forces_', net, '_', preprocess, '.csv']);

axis_to_plot = [3];
uncorrected_pred = uncorrected_pred_forces(:,axis_to_plot+1);
corrected_pred = corrected_pred_forces(:,axis_to_plot+1);
force = force_data(:,axis_to_plot+1);

uncorrected_interp = interp1(uncorrected_pred_forces(:,1), uncorrected_pred, force_data(:,1));
corrected_interp = interp1(corrected_pred_forces(:,1), corrected_pred, force_data(:,1));
force = force(~isnan(corrected_interp));
uncorrected_interp = uncorrected_interp(~isnan(corrected_interp));
corrected_interp = corrected_interp(~isnan(corrected_interp));
fs_loss = rms(uncorrected_interp - force)
loss = rms(corrected_interp - force)
time = force_data(~isnan(corrected_interp), 1);

figure
plot(time, uncorrected_interp, 'r')
title(data)
hold on
plot(time, corrected_interp, 'g') 
plot(time, force, 'b')
legend('uncorrected', 'corrected', 'measured')
title('Force')
