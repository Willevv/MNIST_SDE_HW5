function OneHot = OH(x)
    N = length(x)
    M = zeros(N,10);
    for i = 1:N
        index = x(i)+1;
        M(i, index) = 1;
    end
    OneHot = M;
end