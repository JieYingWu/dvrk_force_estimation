data = 'trocar';
joint_path = ['../data/csv/test/', data, '/no_contact/joints/'];
joint_folder = dir(joint_path);
joint_data = [];
for i = 3%:length(joint_folder)
    temp_data = readmatrix([joint_path, joint_folder(i).name]); 
    joint_data = [joint_data; temp_data];
end

arm_torque = readmatrix(['../results/no_contact/', data, '_torques/arm.csv']);
insertion_torque = readmatrix(['../results/no_contact/', data, '_torques/insertion.csv']);
wrist_torque = readmatrix(['../results/no_contact/', data, '_torques/wrist.csv']);
length = size(joint_data);
length = length(1);
all_torques = zeros(length, 6);
all_jacobians = zeros(length, 36);
all_forces = zeros(length, 6);

all_torques(length - size(arm_torque, 1) + 1:end, 1:2) = arm_torque(:,2:3);
all_torques(length - size(insertion_torque, 1) + 1:end, 3) = insertion_torque(:,2);
all_torques(length - size(wrist_torque, 1) + 1:end, 4:6) = wrist_torque(:,2:4);
measured_torques = joint_data(:,14:19);
axis_to_plot = 3;

figure
%plot(all_torques(:, axis_to_plot), 'r')
%hold on
plot(measured_torques(:, axis_to_plot), 'b')