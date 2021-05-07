clear
data = 'trocar';
contact = 'with_contact';
test_folder = 'test';
net = 'lstm';
preprocess = 'filtered_torque_720s';
seal = 'seal';
file = 1;
pad = 30;
exp = ['exp',num2str(file)];
force_path = ['../data/csv/', test_folder, '/', data, '/', contact, '/', exp, '/sensor/'];
force_data = readmatrix([force_path, 'bag_', num2str(file), '.csv']);

troc_force = readmatrix(['../results/', data, '/', contact, '/', exp, '/', 'lstm_troc_', preprocess, '.csv']);
uncorrected_force = readmatrix(['../results/', data, '/', contact, '/', exp, '/', 'uncorrected_', net, '_' seal, '_', preprocess, '.csv']);
corrected_force = readmatrix(['../results/', data, '/', contact, '/', exp, '/', 'corrected_', net, '_', seal '_', preprocess, '.csv']);
 
troc_force(:,5:6) = -troc_force(:,5:6);
uncorrected_force(:,5:6) = -uncorrected_force(:,5:6);
corrected_force(:,5:6) = -corrected_force(:,5:6);
troc_force(:,5:7) = troc_force(:,5:7)/2.5;
uncorrected_force(:,5:7) = uncorrected_force(:,5:7)/2.5;
corrected_force(:,5:7) = corrected_force(:,5:7)/2.5;

%troc_force = troc_force(troc_force(:,1) > 38 & troc_force(:,1) < 45.9, :);
%uncorrected_force = uncorrected_force(uncorrected_force(:,1) > 38 & uncorrected_force(:,1) < 45.9, :);
%corrected_force = corrected_force(corrected_force(:,1) > 38 & corrected_force(:,1) < 45.9, :);
%force_data = force_data(force_data(:,1) > 38 & force_data(:,1) < 45.9, :);

titles = {'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'};
figure
for axis = 2:7
    subplot(2,3,axis-1);
    plot(troc_force(:,1)-38, troc_force(:,axis),'Color', [0, 0.4, 0])
    title(data)
    hold on
    plot(uncorrected_force(:,1)-38, uncorrected_force(:,axis),'b')
    plot(corrected_force(:,1)-38, corrected_force(:,axis),'m')
    plot(force_data(:,1)-38, force_data(:,axis),'k')
    title(titles{axis-1}, 'FontSize',16);
    xlabel('Time (s)', 'FontSize',12)
    if axis < 5
        ylabel('Force (Nm)', 'FontSize',12)
    else
        ylabel('Torque (Nm)', 'FontSize',12)
    end
    hold off
end
legend('troc', 'seal', 'seal+corr', 'sensor')
    
