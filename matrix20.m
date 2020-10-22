%clc;clear all;
function Matrix20 = matrix20(Input_Matrix)
%%%将输入的400位二进制数排列成20×20的矩阵

%%%初始化
Matrix20 = zeros(20);
for i = 1:20
    for j = 1:20
        Matrix20(i, j) = Input_Matrix(j+(i-1)*20); %输出
    end
end
