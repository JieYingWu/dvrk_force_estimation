fsr_path = ['../data/csv/FSR_calibration/fsr/'];
ati_path = ['../data/csv/FSR_calibration/ati/'];
fsr_folder = dir(ati_path);
ati_data = [];
fsr_data = [];
for i = 3%:length(joint_folder)
    temp_data = readmatrix([ati_path, fsr_folder(i).name]); 
    ati_data = [ati_data; temp_data];
    temp_data = readmatrix([fsr_path, fsr_folder(i).name]); 
    fsr_data = [fsr_data; temp_data];
end

fsr_force = fsr_data(:,2) - min(fsr_data(:,2));
fsr_force = fsr_force/max(fsr_force);
ati_force = get_magnitude(ati_data(:,2:4));
ati_force = ati_force/max(ati_force);

figure
plot(fsr_data(:,1), fsr_force, 'r')
hold on
plot(ati_data(:,1), ati_force, 'b')
legend('FSR', 'ATI')