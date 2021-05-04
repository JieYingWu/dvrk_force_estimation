data = 'trocar';
contact = 'with_contact';
test_folder = 'test';
seal = 'seal';
net = 'lstm';
times = {'60s', '120s', '180s', '240s', '300s', '360s'};%, '420s', '480s'};

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

for file = 0:4
exp = ['exp',num2str(file)];
force_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/sensor/'];
force_data = readmatrix([force_path, 'bag_', num2str(file), '.csv']);

jacobian_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/jacobian/'];
jacobian = readmatrix([jacobian_path, 'interpolated_all_jacobian']);
jacobian = jacobian(pad+1:end, :);

uncorrected_force = readmatrix(['../results/', data, '/', contact, '/', exp, '/', 'uncorrected_', net, '_', seal, '_', preprocess, '.csv']);
corrected_force = readmatrix(['../results/', data, '/', contact, '/', exp, '/', 'corrected_', net, '_',  seal, '_', preprocess, '.csv']);

% len = length(corrected_pred_diff);
% uncorrected_force = zeros([len, 6]);
% corrected_force = zeros([len, 6]);
% for i = 1:len
%     J = inv(reshape(jacobian(i,2:end), 6, 6)')';
%     uncorrected_force(i,:)  = J * (uncorrected_pred_diff(i,2:7)');
%     corrected_force(i,:)  = J * (corrected_pred_diff(i,2:7)');
% end

uncorrected_interp = interp1(uncorrected_force(:,1), uncorrected_force, force_data(:,1));
corrected_interp = interp1(corrected_force(:,1), corrected_force, force_data(:,1));
force = force_data(~isnan(corrected_interp(:,1)),:);
uncorrected_interp = uncorrected_interp(~isnan(corrected_interp(:,1)),:);
corrected_interp = corrected_interp(~isnan(corrected_interp(:,1)),:);

all_uncorrected = [all_uncorrected; uncorrected_interp];
all_corrected = [all_corrected; corrected_interp];
all_force = [all_force; force];
end

all_uncorrected(:,5:7) = all_uncorrected(:,5:7)/-2.5;
all_corrected(:,5:7) = all_corrected(:,5:7)/-2.5;
all_uncorrected(:,7) = -all_uncorrected(:,7);
all_corrected(:,7) = -all_corrected(:,7);

segment_len = 2980; % Empirically determined since this matches 30 s
segments = floor(length(all_force)/segment_len); 
uncorrected_rmse = zeros(segments, 7);
corrected_rmse = zeros(segments, 7);

for i = 1:segments
    start_i = (i-1)*segment_len+1;
    end_i = i*segment_len;
    cur_sensor = all_force(start_i:end_i,:);
    cur_uncorrected = all_uncorrected(start_i:end_i,:);
    cur_corrected = all_corrected(start_i:end_i,:);
    
    uncorrected_rmse(i,:) = rms(cur_sensor-cur_uncorrected);
    corrected_rmse(i,:) = rms(cur_sensor-cur_corrected);    
end

%all_force = all_force(:,2:end);
%fs_loss = all_uncorrected - all_force;
fs_rms = mean(uncorrected_rmse, 1);
fs_std = std(uncorrected_rmse, 1);
all_fs_rms = [all_fs_rms; fs_rms];
all_fs_std = [all_fs_std; fs_std];

%loss = all_corrected - all_force;
corr_rms = mean(corrected_rmse, 1);
corr_std = std(corrected_rmse, 1);
all_rms = [all_rms; corr_rms];
all_std = [all_std; corr_std];

end

