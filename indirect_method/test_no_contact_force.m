data = 'trocar';
contact = 'with_contact';
test_folder = 'test';
net = 'lstm';
times = {'600'};%, '360', '420s', '480s'};

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

all_troc = readmatrix(['../results/', data, '/no_contact/test/lstm_troc_', preprocess, 's.csv']);
all_troc = all_troc(pad:end,:);
base_all_uncorrected = readmatrix(['../results/', data, '/no_contact/test/uncorrected_', net, '_base_', preprocess, 's.csv']);
base_all_corrected = readmatrix(['../results/', data, '/no_contact/test/corrected_', net, '_base_', preprocess, 's.csv']);
seal_all_uncorrected = readmatrix(['../results/', data, '/no_contact/test/uncorrected_', net, '_seal_', preprocess, 's.csv']);
seal_all_corrected = readmatrix(['../results/', data, '/no_contact/test/corrected_', net, '_seal_', preprocess, 's.csv']);

offset = repmat([0,0,0.075], [length(all_troc),1]);
all_troc(:,5:7) = all_troc(:,5:7) + cross(offset, all_troc(:,2:4), 2);
all_troc(:,5:6) = -all_troc(:,5:6);

offset = repmat([0,0,0.075], [length(base_all_uncorrected),1]);
base_all_uncorrected(:,5:7) = base_all_uncorrected(:,5:7) + cross(offset, base_all_uncorrected(:,2:4), 2);
base_all_corrected(:,5:7) = base_all_corrected(:,5:7) + cross(offset, base_all_corrected(:,2:4), 2);
seal_all_uncorrected(:,5:7) = seal_all_uncorrected(:,5:7) + cross(offset, seal_all_uncorrected(:,2:4), 2);
seal_all_corrected(:,5:7) = seal_all_corrected(:,5:7) + cross(offset, seal_all_corrected(:,2:4), 2);

troc_rms = rms(all_troc);
troc_std = std(abs(all_troc), 1);

base_fs_rms = rms(base_all_uncorrected, 1);
base_fs_std = std(abs(base_all_uncorrected), 1);
base_rms = rms(base_all_corrected, 1); 
base_std = std(abs(base_all_corrected), 1);

seal_fs_rms = rms(seal_all_uncorrected, 1);
seal_fs_std = std(abs(seal_all_uncorrected), 1);

seal_rms = rms(seal_all_corrected, 1);
seal_std = std(abs(seal_all_corrected), 1);

end
