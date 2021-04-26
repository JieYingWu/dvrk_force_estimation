clear
data = 'trocar';
contact = 'with_contact';
test_folder = 'test';
net = 'ff';
times = {'30s', '60s', '90s', '120s', '150s', '180s'};

%file = 3;
all_fs_rms = [];
all_fs_std = [];
all_rms = [];
all_std = [];


all_uncorrected = [];
all_corrected = [];
all_force = [];
for t = 1:6
preprocess = ['filtered_torque_', times{t}];
for file = 0:3
exp = ['exp',num2str(file)];
joint_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/joints/'];
force_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/sensor/'];

joint_folder = dir(joint_path);
joint_data = readmatrix([joint_path, 'interpolated_all_joints']);
force_data = readmatrix([force_path, joint_folder(3).name]);

uncorrected_pred_forces = readmatrix(['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/', 'uncorrected_forces_', net, '_', preprocess, '.csv']);
corrected_pred_forces = readmatrix(['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/', 'corrected_forces_', net, '_', preprocess, '.csv']);

uncorrected_interp = interp1(uncorrected_pred_forces(:,1), uncorrected_pred_forces, force_data(:,1));
corrected_interp = interp1(corrected_pred_forces(:,1), corrected_pred_forces, force_data(:,1));
force = force_data(~isnan(corrected_interp(:,1)),:);
uncorrected_interp = uncorrected_interp(~isnan(corrected_interp(:,1)),:);
corrected_interp = corrected_interp(~isnan(corrected_interp(:,1)),:);
all_uncorrected = [all_uncorrected; uncorrected_interp];
all_corrected = [all_corrected; corrected_interp];
all_force = [all_force; force];
end

fs_loss = all_uncorrected - all_force;
fs_rms = rms(fs_loss, 1);
fs_std = std(fs_loss, 1);
all_fs_rms = [all_fs_rms; fs_rms];
all_fs_std = [all_fs_std; fs_std];

loss = all_corrected - all_force;
corr_rms = rms(loss, 1);
corr_std = std(loss, 1);
all_rms = [all_rms; corr_rms];
all_std = [all_std; corr_std];

end

