force_path = '../data/csv/test/free_space/force_sensor/sensor/';
joint_path = '../data/csv/test/free_space/force_sensor/joints/';
force_folder = dir(force_path);
if ~exist('force_data')
    force_data = [];
    joint_data = [];
    for i = 3:length(force_folder)
        data = readmatrix([force_path, force_folder(i).name]); 
        force_data = [force_data; data];
        data = readmatrix([joint_path, force_folder(i).name]); 
        joint_data = [joint_data; data];
    end

    arm_torque = readmatrix('../results/free_space_torques/arm.csv');
    insertion_torque = readmatrix('../results/free_space_torques/insertion.csv');
    wrist_torque = readmatrix('../results/free_space_torques/wrist.csv');
end


length = size(joint_data);
length = length(1);
all_torques = zeros(length, 6);
all_jacobians = zeros(length, 36);
all_forces = zeros(length, 6);

all_torques(length - size(arm_torque, 1) + 1:end, 1:2) = arm_torque(:,2:3);
all_torques(length - size(insertion_torque, 1) + 1:end, 3) = insertion_torque(:,2);
all_torques(length - size(wrist_torque, 1) + 1:end, 4:6) = wrist_torque(:,2:4);
all_jacobians(length - size(wrist_torque, 1) + 1:end, :) = wrist_torque(:, 5:end);

for i = 1:length
    J = inv(reshape(all_jacobians(i,:), 6, 6)')';
    all_forces(i,:)  = J * all_torques(i,:)';
end

axis_to_plot = 3;
figure
plot(joint_data(:,1), all_forces(:,axis_to_plot), 'r')
hold on
plot(force_data(:,1), force_data(:,axis_to_plot+1), 'b')