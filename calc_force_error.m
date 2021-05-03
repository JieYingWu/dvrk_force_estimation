data = 'trocar';
contact = 'with_contact';
test_folder = 'test';
seal = 'seal';
net = 'ff';
times = {'60s', '120s', '180s', '240s', '300s', '360s', '420s', '480s'};

%file = 3;
all_fs_rms = [];
all_fs_std = [];
all_rms = [];
all_std = [];
pad = 30;


for t = 1:length(times)
preprocess = ['filtered_torque_', times{t}];
all_uncorrected = [];
all_corrected = [];
all_force = [];

for file = 0:3
exp = ['exp',num2str(file)];
joint_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/joints/'];
force_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/sensor/'];

joint_folder = dir(joint_path);
joint_data = readmatrix([joint_path, 'interpolated_all_joints']);
force_data = readmatrix([force_path, joint_folder(3).name]);

jacobian_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/jacobian/'];
jacobian = readmatrix([jacobian_path, 'interpolated_all_jacobian']);
jacobian = jacobian(pad+1:end, :);

uncorrected_pred_diff = readmatrix(['../results/', data, '/', contact, '/', exp, '/', 'uncorrected_', net, '_', seal, '_', preprocess, '.csv']);
corrected_pred_diff = readmatrix(['../results/', data, '/', contact, '/', exp, '/', 'corrected_', net, '_',  seal, '_', preprocess, '.csv']);
filtered_torque = joint_data(:,14:19);

len = length(corrected_pred_diff);
uncorrected_force = zeros([len, 6]);
corrected_force = zeros([len, 6]);
for i = 1:len
    J = inv(reshape(jacobian(i,2:end), 6, 6)')';
    uncorrected_force(i,:)  = J * (uncorrected_pred_diff(i,2:7)');
    corrected_force(i,:)  = J * (corrected_pred_diff(i,2:7)');
end

uncorrected_interp = interp1(uncorrected_pred_diff(:,1), uncorrected_force, force_data(:,1));
corrected_interp = interp1(corrected_pred_diff(:,1), corrected_force, force_data(:,1));
force = force_data(~isnan(corrected_interp(:,1)),:);
uncorrected_interp = uncorrected_interp(~isnan(corrected_interp(:,1)),:);
corrected_interp = corrected_interp(~isnan(corrected_interp(:,1)),:);
all_uncorrected = [all_uncorrected; uncorrected_interp];
all_corrected = [all_corrected; corrected_interp];
all_force = [all_force; force];
end

all_uncorrected(:,4:6) = all_uncorrected(:,4:6)/-2.6931;
all_corrected(:,4:6) = all_corrected(:,4:6)/-2.6931;
all_uncorrected(:,6) = -all_uncorrected(:,6);
all_corrected(:,6) = -all_corrected(:,6);


all_force = all_force(:,2:end);
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

