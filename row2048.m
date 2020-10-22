function D = row2048(A)
%%%将输入的16×16的十进制矩阵转换为1×2048的二进制行数组
A = floor(A);
B = dec2bin(A', 8);
for i = 1:256
    for j = 1:8
        C(i,j) = str2double(B(i, j));
    end
end

D = zeros(1, 2048);
for i = 1:256
    for j = 1:8
        D(1, j+(i-1)*8) = C(i, j); 
    end
end