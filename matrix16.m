%clc;clear all;
function Matrix16 = matrix16(Input_Matrix)
%%%将输入的2048位二进制数转化为[0,255]的十进制数，并重新排列成16×16的矩阵

%%%初始化
M_Matrix = zeros(size(2048/8, 8));
Str_Matrix = char(zeros(size(2048/8, 8)));
Matrix16 = zeros(16);

for i = 1:2048/8
    for j = 1:8
        M_Matrix(i, j) = Input_Matrix(j+(i-1)*8); %Input_Matrix为输入的二进制数据，M_Matrix为中间状态矩阵
    end
end
M_Matrix = num2str(M_Matrix);
M_Matrix(isspace(M_Matrix)) = [];  %转化为字符串
for i = 1:2048/8
    for j = 1:8
        Str_Matrix(i, j) = M_Matrix(j+(i-1)*8); %重新排列成256×8的形式
    end
end
K = bin2dec(Str_Matrix); %转化为十进制

for i = 1:16
    for j = 1:16
        Matrix16(i, j) = K(j+(i-1)*16); %输出
    end
end


