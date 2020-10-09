data = 'trocar';
exp = 'exp0';
test = 'test';
if ~exist('joint_data', 'var')
joint_path = ['../data/csv/test/', data, '/no_contact/', exp, '/joints/'];
joint_folder = dir(joint_path);
joint_data = [];
for i = 3%:length(joint_folder)
    temp_data = readmatrix([joint_path, joint_folder(i).name]); 
    joint_data = [joint_data; temp_data];
end
end

arm_torque = readmatrix(['../results/', test, '/no_contact/', data, '/', exp, '/arm.csv']);
insertion_torque = readmatrix(['../results/', test, '/no_contact/', data, '/', exp, '/insertion.csv']);
wrist_torque = readmatrix(['../results/', test, '/no_contact/', data, '/', exp, '/wrist.csv']);
length = size(joint_data);
length = length(1);
all_torques = zeros(length, 6);

all_torques(length - size(arm_torque, 1) + 1:end, 1:2) = arm_torque(:,2:3);
all_torques(length - size(insertion_torque, 1) + 1:end, 3) = insertion_torque(:,2);
all_torques(length - size(wrist_torque, 1) + 1:end, 4:6) = wrist_torque(:,2:4);
measured_torques = joint_data(:,14:19);
axis_to_plot = 3;

figure
plot(measured_torques(:, axis_to_plot), 'b')
hold on
plot(all_torques(:, axis_to_plot), 'r')
legend('measured','pred')
title('Torque')
