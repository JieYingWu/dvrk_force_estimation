clear
data = 'trocar';
contact = 'with_contact';
test_folder = 'test';
net = 'lstm';
preprocess = 'filtered_torque_480s';
seal = 'seal';
file = 1;
pad = 30;
exp = ['exp',num2str(file)];
joint_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/joints/'];
jacobian_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/jacobian/'];
force_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/sensor/'];

joint_folder = dir(joint_path);
joint_data = readmatrix([joint_path, 'interpolated_all_joints']);
jacobian = readmatrix([jacobian_path, 'interpolated_all_jacobian']);
jacobian = jacobian(pad+1:end, :);
fs_pred = readmatrix(['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/', net, '_seal_pred_filtered_torque.csv']);
force_data = readmatrix([force_path, joint_folder(3).name]);

uncorrected_pred_diff = readmatrix(['../results/', data, '/', contact, '/', exp, '/', 'uncorrected_', net, '_' seal, '_', preprocess, '.csv']);
corrected_pred_diff = readmatrix(['../results/', data, '/', contact, '/', exp, '/', 'corrected_', net, '_', seal '_', preprocess, '.csv']);
filtered_torque = joint_data(:,14:19);

len = length(corrected_pred_diff);
uncorrected_force = zeros([len, 6]);
corrected_force = zeros([len, 6]);
for i = 1:len
    J = inv(reshape(jacobian(i,2:end), 6, 6)')';
    uncorrected_force(i,:)  = J * (uncorrected_pred_diff(i,2:7)');
    corrected_force(i,:)  = J * (corrected_pred_diff(i,2:7)');
end

uncorrected_force(:,4:6) = uncorrected_force(:,4:6)/-2.6931;
corrected_force(:,4:6) = corrected_force(:,4:6)/-2.6931;
uncorrected_force(:,6) = -uncorrected_force(:,6);
corrected_force(:,6) = -corrected_force(:,6);


axis_to_plot = [3];
uncorrected_pred = uncorrected_force(:,axis_to_plot);
corrected_pred = corrected_force(:,axis_to_plot);
force = force_data(:,axis_to_plot+1);

uncorrected_interp = interp1(uncorrected_pred_diff(:,1), uncorrected_pred, force_data(:,1));
corrected_interp = interp1(corrected_pred_diff(:,1), corrected_pred, force_data(:,1));
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

figure
plot(joint_data(:,1), joint_data(:,axis_to_plot+13))
hold on
plot(fs_pred(:,1), fs_pred(:,axis_to_plot+1))
legend('Measured', 'fs')
hold off
