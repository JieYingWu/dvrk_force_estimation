data = 'trocar';
contact = 'no_contact';
test_folder = 'test';
seal = 'base';
net = 'lstm';
times = {'60s', '120s', '180s', '240s', '300s'};%, '360s'};%, '420s', '480s'};

all_fs_rms = [];
all_fs_std = [];
all_rms = [];
all_std = [];
pad = 30;

for t = 1:length(times)
preprocess = ['filtered_torque_', times{t}];
joint_path = ['../data/csv/', test_folder, '/', data, '/no_contact/joints/'];


joint_folder = dir(joint_path);
joint_data = readmatrix([joint_path, 'interpolated_all_joints']);
measured = joint_data(:, 14:19);
fs_pred = readmatrix(['../data/csv/test/trocar/no_contact/', net, '_', seal, '_pred_filtered_torque.csv']);
fs_pred = fs_pred(:,2:end);
pred = readmatrix(['../results/trocar/no_contact/', 'torque_', net, '_', seal, '_', preprocess, '.csv']);
pred = pred(:, 2:end);

fs_loss = fs_pred - measured(1:length(fs_pred), :);
fs_rms = rms(fs_loss, 1);
fs_std = std(fs_loss, 1);
all_fs_rms = [all_fs_rms; fs_rms];
all_fs_std = [all_fs_std; fs_std];

measured = joint_data(pad+1:end, 14:19);
loss = pred - measured(1:length(pred), :);
corr_rms = rms(loss, 1);
corr_std = std(loss, 1);
all_rms = [all_rms; corr_rms];
all_std = [all_std; corr_std];

end

