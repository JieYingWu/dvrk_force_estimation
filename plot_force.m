data = 'free_space';
contact = 'with_contact';
test_folder = 'test_7dof';
file = 0;
exp = ['exp',num2str(file)];
force_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/sensor/'];
joint_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/joints/'];

joint_folder = dir(joint_path);
force_data = readmatrix([force_path, joint_folder(3).name]);
joint_data = readmatrix([joint_path, joint_folder(3).name]); 

%data = 'corrected';
path = ['../results/', test_folder, '/', contact, '/', data, '/', exp];
arm_torque = readmatrix([path, '/arm.csv']);
insertion_torque = readmatrix([path, '/insertion.csv']);
wrist_torque = readmatrix([path, '/wrist.csv']);

length = size(joint_data);
length = length(1);
all_torques = zeros(length, 6);
all_jacobians = zeros(length, 36);
all_forces = zeros(length, 6);

all_torques(length - size(arm_torque, 1) + 1:end, 1:2) = arm_torque(:,2:3);
all_torques(length - size(insertion_torque, 1) + 1:end, 3) = insertion_torque(:,2);
all_torques(length - size(wrist_torque, 1) + 1:end, 5:6) = wrist_torque(:,2:3);
measured_torques = joint_data(:,14:19);
all_torques = measured_torques - all_torques;
all_jacobians(length - size(arm_torque, 1) + 1:end, :) = arm_torque(:, 4:end);

for i = 1:length
    J = inv(reshape(all_jacobians(i,:), 6, 6))';
    all_forces(i,:)  = J * all_torques(i,:)';
end

axis_to_plot = [3];
predicted = all_forces(:,axis_to_plot);
%predicted(predicted > 0) = 0;

figure
plot(joint_data(:, 1), predicted, 'r')
title(data)
hold on
plot(force_data(:, 1), force_data(:,axis_to_plot+1), 'b')
legend('pred', 'measured')
title('Force')

