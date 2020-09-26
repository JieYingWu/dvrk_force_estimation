fsr_path = ['../data/csv/FSR_calibration/fsr/'];
ati_path = ['../data/csv/FSR_calibration/ati/'];
fsr_folder = dir(ati_path);
ati_data = [];
fsr_data = [];
for i = 3%:length(joint_folder)
    temp_data = readmatrix([fsr_path, fsr_folder(i).name]); 
    ati_data = [ati_data; temp_data];
    temp_data = readmatrix([ati_path, fsr_folder(i).name]); 
    fsr_data = [fsr_data; temp_data];
end