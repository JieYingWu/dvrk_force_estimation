function print_err(all_rms, all_std, row)

if row
    
sprintf("%.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f)", all_rms(5, 1), all_std(5, 1), ...
all_rms(5, 2), all_std(5, 2), all_rms(5, 3), all_std(5, 3), all_rms(5, 4), all_std(5, 4), all_rms(5, 5), all_std(5, 5), all_rms(5, 6), all_std(5, 6)) 
    
else 
    
for col = 2:7
sprintf("%.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f)", all_rms(1, col), all_std(1, col), ...
all_rms(2, col), all_std(2, col), all_rms(3, col), all_std(3, col), all_rms(4, col), all_std(4, col))
end

end

end
