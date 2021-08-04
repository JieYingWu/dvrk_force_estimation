seal = rms(seal_rms(:,2:4), 2);
seal_fs = rms(seal_fs_rms(:,2:4), 2);
base = rms(base_rms(:,2:4), 2);
base_fs = rms(base_fs_rms(:,2:4), 2);
troc = rms(troc_rms(:,2:4), 2);

x = [2,4,6,8,10,12,14,16,18];
figure
plot(x, troc, 'k-*')
hold on
%plot(x, base_fs, 'r-*')
%plot(x, seal_fs, 'b-*')
plot(x, base, 'r-*')
plot(x, seal, 'b-*')
ylabel('Mean RMSE over Fx, Fy, Fz (N)')
xlabel('Length of training set (min)')
legend('Troc', 'Base+Corr', 'Seal+Corr')
hold off