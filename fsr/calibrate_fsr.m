fsr_path = ['../data/csv/FSR_calibration/fsr/'];
ati_path = ['../data/csv/FSR_calibration/ati/'];
fsr_folder = dir(ati_path);
ati_data = [];
fsr_data = [];
for i = 3:4%:length(joint_folder)
    ati_data = readmatrix([ati_path, fsr_folder(i).name]); 
    fsr_data = readmatrix([fsr_path, fsr_folder(i).name]); 

    ati_force = -1*ati_data(:,4);
    

fsr_force = interp1(fsr_data(:,1), fsr_data(:,2), ati_data(:,1));
ati_force = ati_force(~isnan(fsr_force));
fsr_force = fsr_force(~isnan(fsr_force));
c = polyfit(fsr_force, ati_force, 1)

fsr_force = c(1) * fsr_data(:,2) + c(2);
figure
plot(fsr_data(:,1), fsr_force, 'r')
hold on
plot(ati_data(:,1), -1*ati_data(:,4), 'b')
legend('FSR', 'ATI')

end
