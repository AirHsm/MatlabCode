function array = cell2array1d(cell)

n = max(size(cell));
array = zeros(n,1);
for i = 1:n
    array(i) = cell{i};
end