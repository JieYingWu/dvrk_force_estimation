% function print_err(all_rms, all_std, table_4)
% 
% if row
% sprintf("%.3f (%.3f) & %.3f (%.3f) & %.3f (%.3f) & %.3f (%.3f) & %.3f (%.3f) & %.3f (%.3f)", all_rms(5, 1), all_std(5, 1), ...
% all_rms(5, 2), all_std(5, 2), all_rms(5, 3), all_std(5, 3), all_rms(5, 4), all_std(5, 4), all_rms(5, 5), all_std(5, 5), all_rms(5, 6), all_std(5, 6)) 
%     
% else 
%     
% for col = 2:7
% sprintf("%.3f (%.3f) & %.3f (%.3f) & %.3f (%.3f) & %.3f (%.3f)", all_rms(1, col), all_std(1, col), ...
% all_rms(2, col), all_std(2, col), all_rms(3, col), all_std(3, col), all_rms(4, col), all_std(4, col))
% end
% 
% end
% 
% end

if false
for col = 2:7
for row = 2:2:8
    sprintf("%.3f (%.3f) & %.3f (%.3f) & %.3f (%.3f)", troc_rms(row, col), troc_std(row, col), ...
    base_rms(row, col), base_std(row, col), seal_rms(row, col), seal_std(row, col))
end    
end
end

if false
for col = 2:7
for row = 1
    sprintf("%.3f (%.3f) & %.3f (%.3f)", base_fs_rms(row, col), base_fs_std(row, col), seal_fs_rms(row, col), seal_fs_std(row, col))
end    
end
end

if true
for col = 2:7
for row = 1
    sprintf("%.3f (%.3f) & %.3f (%.3f) & %.3f (%.3f)& %.3f (%.3f) & %.3f (%.3f)", troc_rms(row, col), troc_std(row, col), ...
        base_fs_rms(row, col), base_fs_std(row, col), seal_fs_rms(row, col), seal_fs_std(row, col), base_rms(row, col), base_std(row, col), seal_rms(row, col), seal_std(row, col))
end    
end
end