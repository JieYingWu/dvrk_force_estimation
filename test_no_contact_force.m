data = 'trocar';
contact = 'with_contact';
test_folder = 'test';
seal = 'seal';
net = 'lstm';
times = {'60s', '120s', '180s', '240s', '300s'};%, '360', '420s', '480s'};

%file = 3;
all_troc_rms = [];
all_troc_std = [];
all_fs_rms = [];
all_fs_std = [];
all_rms = [];
all_std = [];
pad = 30;


for t = 1:length(times)
preprocess = ['filtered_torque_', times{t}];
all_troc = [];
all_uncorrected = [];
all_corrected = [];
all_force = [];

troc_force = readmatrix(['../results/', data, '/no_contact/test/', 'lstm_troc_', preprocess, '.csv']);
uncorrected_force = readmatrix(['../results/', data, '/no_contact/test/', 'uncorrected_', net, '_', seal, '_', preprocess, '.csv']);
corrected_force = readmatrix(['../results/', data, '/no_contact/test/', 'corrected_', net, '_',  seal, '_', preprocess, '.csv']);
force_data = zeros(size(uncorrected_force));
force_data(:,1) = corrected_force(:,1);

troc_interp = interp1(troc_force(:,1), troc_force, force_data(:,1));
uncorrected_interp = interp1(uncorrected_force(:,1), uncorrected_force, force_data(:,1));
corrected_interp = interp1(corrected_force(:,1), corrected_force, force_data(:,1));

force = force_data(~isnan(corrected_interp(:,1)),:);
troc_interp = troc_interp(~isnan(corrected_interp(:,1)),:);
uncorrected_interp = uncorrected_interp(~isnan(corrected_interp(:,1)),:);
corrected_interp = corrected_interp(~isnan(corrected_interp(:,1)),:);

all_troc = [all_troc; troc_interp];
all_uncorrected = [all_uncorrected; uncorrected_interp];
all_corrected = [all_corrected; corrected_interp];
all_force = [all_force; force];

all_troc(:, 5:7) = all_troc(:,5:7)/2.5;
all_uncorrected(:,5:7) = all_uncorrected(:,5:7)/2.5;
all_corrected(:,5:7) = all_corrected(:,5:7)/2.5;
all_troc(:,5:6) = -all_troc(:,5:6);
all_uncorrected(:,5:6) = -all_uncorrected(:,5:6);
all_corrected(:,5:6) = -all_corrected(:,5:6);

segment_len = 2980; % Empirically determined since this matches 30 s
segments = floor(length(all_force)/segment_len); 
troc_rmse = zeros(segments, 7);
uncorrected_rmse = zeros(segments, 7);
corrected_rmse = zeros(segments, 7);

for i = 1:segments
    start_i = (i-1)*segment_len+1;
    end_i = i*segment_len;
    cur_sensor = all_force(start_i:end_i,:);
    cur_troc = all_troc(start_i:end_i,:);
    cur_uncorrected = all_uncorrected(start_i:end_i,:);
    cur_corrected = all_corrected(start_i:end_i,:);
    
    troc_rmse(i,:) = rms(cur_sensor-cur_troc);
    uncorrected_rmse(i,:) = rms(cur_sensor-cur_uncorrected);
    corrected_rmse(i,:) = rms(cur_sensor-cur_corrected);    
end

troc_rms = mean(troc_rmse, 1);
troc_std = std(troc_rmse, 1);
all_troc_rms = [all_troc_rms; troc_rms];
all_troc_std = [all_troc_std; troc_std];

fs_rms = mean(uncorrected_rmse, 1);
fs_std = std(uncorrected_rmse, 1);
all_fs_rms = [all_fs_rms; fs_rms];
all_fs_std = [all_fs_std; fs_std];

corr_rms = mean(corrected_rmse, 1);
corr_std = std(corrected_rmse, 1);
all_rms = [all_rms; corr_rms];
all_std = [all_std; corr_std];

end
