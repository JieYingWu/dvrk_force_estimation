function y = get_magnitude(x)
    y = x(:,1).^2 + x(:,2).^2 + x(:,3).^2;
    y = sqrt(y);
end
