data = 'free_space';
contact = 'with_contact';
force_path = ['../data/csv/test/', data, '/', contact, '/sensor/'];
joint_path = ['../data/csv/test/', data, '/', contact, '/joints/'];
joint_folder = dir(joint_path);
force_data = [];
joint_data = [];
for i = 3%:length(joint_folder)
    temp_data = readmatrix([force_path, joint_folder(i).name]); 
    force_data = [force_data; temp_data];
    temp_data = readmatrix([joint_path, joint_folder(i).name]); 
    joint_data = [joint_data; temp_data];
end

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
all_torques(length - size(wrist_torque, 1) + 1:end, 4:6) = wrist_torque(:,2:4);
measured_torques = joint_data(:,14:19);
all_torques = measured_torques - all_torques;
all_jacobians(length - size(wrist_torque, 1) + 1:end, :) = wrist_torque(:, 5:end);

for i = 1:length
    J = inv(reshape(all_jacobians(i,:), 6, 6))';
    all_forces(i,:)  = J * all_torques(i,:)';
end

axis_to_plot = [3];
predicted = all_forces(:,axis_to_plot);
predicted(predicted > 0) = 0;

figure
plot(joint_data(:,1), predicted, 'r')
title(data)
hold on
plot(force_data(:,1), force_data(:,axis_to_plot+1), 'b')