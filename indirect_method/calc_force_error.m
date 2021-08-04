data = 'trocar';
contact = 'with_contact';
test_folder = 'test';
net = 'lstm';
times = {'120', '240', '360', '480', '600', '720', '840', '960', '1080'};%{'60s', '120s', '180s', '240s', '300s'};%, '360', '420s', '480s'};

troc_rms = [];
troc_std = [];
base_fs_rms = [];
base_fs_std = [];
base_rms = [];
base_std = [];
seal_fs_rms = [];
seal_fs_std = [];
seal_rms = [];
seal_std = [];
pad = 30;

for t = 1:length(times)
preprocess = ['filtered_torque_', times{t}];
all_troc = [];
base_all_uncorrected = [];
base_all_corrected = [];
seal_all_uncorrected = [];
seal_all_corrected = [];
all_force = [];

for file = 0:4
exp = ['exp',num2str(file)];
force_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/sensor/'];
force_data = readmatrix([force_path, 'bag_', num2str(file), '.csv']);

troc_force = readmatrix(['../results/', data, '/', contact, '/', exp, '/', 'lstm_troc_', preprocess, 's.csv']);
base_uncorrected_force = readmatrix(['../results/', data, '/', contact, '/', exp, '/', 'uncorrected_', net, '_base_', preprocess, 's.csv']);
base_corrected_force = readmatrix(['../results/', data, '/', contact, '/', exp, '/', 'corrected_', net, '_base_', preprocess, 's.csv']);
seal_uncorrected_force = readmatrix(['../results/', data, '/', contact, '/', exp, '/', 'uncorrected_', net, '_seal_', preprocess, 's.csv']);
seal_corrected_force = readmatrix(['../results/', data, '/', contact, '/', exp, '/', 'corrected_', net, '_seal_', preprocess, 's.csv']);

troc_interp = interp1(troc_force(:,1), troc_force, force_data(:,1));
base_uncorrected_interp = interp1(base_uncorrected_force(:,1), base_uncorrected_force, force_data(:,1));
base_corrected_interp = interp1(base_corrected_force(:,1), base_corrected_force, force_data(:,1));
seal_uncorrected_interp = interp1(seal_uncorrected_force(:,1), seal_uncorrected_force, force_data(:,1));
seal_corrected_interp = interp1(seal_corrected_force(:,1), seal_corrected_force, force_data(:,1));

force = force_data(~isnan(base_corrected_interp(:,1)),:);
troc_interp = troc_interp(~isnan(base_corrected_interp(:,1)),:);
base_uncorrected_interp = base_uncorrected_interp(~isnan(base_uncorrected_interp(:,1)),:);
base_corrected_interp = base_corrected_interp(~isnan(base_corrected_interp(:,1)),:);
seal_uncorrected_interp = seal_uncorrected_interp(~isnan(seal_uncorrected_interp(:,1)),:);
seal_corrected_interp = seal_corrected_interp(~isnan(seal_corrected_interp(:,1)),:);

all_troc = [all_troc; troc_interp];
base_all_uncorrected = [base_all_uncorrected; base_uncorrected_interp];
base_all_corrected = [base_all_corrected; base_corrected_interp];
seal_all_uncorrected = [seal_all_uncorrected; seal_uncorrected_interp];
seal_all_corrected = [seal_all_corrected; seal_corrected_interp];
all_force = [all_force; force];
end

offset = repmat([0,0,0.075], [length(all_troc),1]);
all_troc(:,5:7) = all_troc(:,5:7) + cross(offset, all_troc(:,2:4), 2);
all_troc(:,5:6) = -all_troc(:,5:6);

offset = repmat([0,0,0.075], [length(base_all_uncorrected),1]);
base_all_uncorrected(:,5:7) = base_all_uncorrected(:,5:7) + cross(offset, base_all_uncorrected(:,2:4), 2);
base_all_corrected(:,5:7) = base_all_corrected(:,5:7) + cross(offset, base_all_corrected(:,2:4), 2);
seal_all_uncorrected(:,5:7) = seal_all_uncorrected(:,5:7) + cross(offset, seal_all_uncorrected(:,2:4), 2);
seal_all_corrected(:,5:7) = seal_all_corrected(:,5:7) + cross(offset, seal_all_corrected(:,2:4), 2);

base_all_uncorrected(:,5:6) = -base_all_uncorrected(:,5:6);
base_all_corrected(:,5:6) = -base_all_corrected(:,5:6);
seal_all_uncorrected(:,5:6) = -seal_all_uncorrected(:,5:6);
seal_all_corrected(:,5:6) = -seal_all_corrected(:,5:6);

segment_len = 3000; 
segments = floor(length(all_force)/segment_len); 
troc_rmse = zeros(segments, 7);
base_uncorrected_rmse = zeros(segments, 7);
base_corrected_rmse = zeros(segments, 7);
seal_uncorrected_rmse = zeros(segments, 7);
seal_corrected_rmse = zeros(segments, 7);

for i = 1:segments
    start_i = (i-1)*segment_len+1;
    end_i = i*segment_len;
    cur_sensor = all_force(start_i:end_i,:);
    cur_troc = all_troc(start_i:end_i,:);
    base_cur_uncorrected = base_all_uncorrected(start_i:end_i,:);
    base_cur_corrected = base_all_corrected(start_i:end_i,:);
    seal_cur_uncorrected = seal_all_uncorrected(start_i:end_i,:);
    seal_cur_corrected = seal_all_corrected(start_i:end_i,:);
      
    troc_rmse(i,:) = rms(cur_sensor-cur_troc);
    base_uncorrected_rmse(i,:) = rms(cur_sensor-base_cur_uncorrected);
    base_corrected_rmse(i,:) = rms(cur_sensor-base_cur_corrected);    
    seal_uncorrected_rmse(i,:) = rms(cur_sensor-seal_cur_uncorrected);
    seal_corrected_rmse(i,:) = rms(cur_sensor-seal_cur_corrected);    
end

troc_cur_rms = mean(troc_rmse, 1);
troc_cur_std = std(troc_rmse, 1);
troc_rms = [troc_rms; troc_cur_rms];
troc_std = [troc_std; troc_cur_std];

base_fs_cur_rms = mean(base_uncorrected_rmse, 1);
base_fs_cur_std = std(base_uncorrected_rmse, 1);
base_fs_rms = [base_fs_rms; base_fs_cur_rms];
base_fs_std = [base_fs_std; base_fs_cur_std];

base_cur_rms = mean(base_corrected_rmse, 1); 
base_cur_std = std(base_corrected_rmse, 1);
base_rms = [base_rms; base_cur_rms];
base_std = [base_std; base_cur_std];

seal_fs_cur_rms = mean(seal_uncorrected_rmse, 1);
seal_fs_cur_std = std(seal_uncorrected_rmse, 1);
seal_fs_rms = [seal_fs_rms; seal_fs_cur_rms];
seal_fs_std = [seal_fs_std; seal_fs_cur_std];

seal_cur_rms = mean(seal_corrected_rmse, 1);
seal_cur_std = std(seal_corrected_rmse, 1);
seal_rms = [seal_rms; seal_cur_rms];
seal_std = [seal_std; seal_cur_std];

end

