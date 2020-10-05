data = 'free_space';
contact = 'with_contact';
test_folder = 'test_7dof';
force_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/sensor/'];
joint_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/joints/'];

joint_folder = dir(joint_path);
force_data = [];
joint_data = [];
files = size(joint_folder);
force_lengths = zeros(1,files(1)-2);
joint_lengths = zeros(1,files(1)-2);
for i = 3:files(1)
    temp_data = readmatrix([force_path, joint_folder(i).name]); 
    force_data = [force_data; temp_data];
    temp_size = size(temp_data);
    temp_data = readmatrix([joint_path, joint_folder(i).name]); 
    force_lengths(i-2) = temp_size(1);
    joint_data = [joint_data; temp_data];
    temp_size = size(temp_data);
    joint_lengths(i-2) = temp_size(1);
end

%data = 'corrected';
arm_torque = readmatrix(['../results/', contact, '/', data, '_torques/arm.csv']);
insertion_torque = readmatrix(['../results/', contact, '/', data, '_torques/insertion.csv']);
wrist_torque = readmatrix(['../results/', contact, '/', data, '_torques/wrist.csv']);

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

axis_to_plot = [2];
predicted = all_forces(:,axis_to_plot);
predicted(predicted > 0) = 0;
file_to_plot = 2;
start_index_force = sum(force_lengths(1:file_to_plot-1))+1;
end_index_force = sum(force_lengths(1:file_to_plot));
start_index_joint = sum(joint_lengths(1:file_to_plot-1))+1;
end_index_joint = sum(joint_lengths(1:file_to_plot));

figure
plot(joint_data(start_index_joint:end_index_joint, 1), predicted(start_index_joint:end_index_joint), 'r')
title(data)
hold on
plot(force_data(start_index_force:end_index_force, 1), force_data(start_index_force:end_index_force,axis_to_plot+1), 'b')
legend('pred', 'measured')
title('Force')

