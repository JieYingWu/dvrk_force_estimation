data = readmatrix('pred_trocar.csv');

measured = data(:,1:6);
free_space = data(:, 7:12);
trocar = data(:, 13:18);

range = 5000:6999;
figure

for joint = 1:6 
    subplot(6, 1, joint)
    plot(measured(range,joint), 'k')
    hold on
    plot(free_space(range,joint), 'b')
    plot(trocar(range,joint), 'r')
    title(['Joint ', int2str(joint)])
    legend('Measured', 'Free space network', 'Trocar network')
    hold off

end