%clc;clear all;
function Matrix20 = matrix20(Input_Matrix)
%%%�������400λ�����������г�20��20�ľ���

%%%��ʼ��
Matrix20 = zeros(20);
for i = 1:20
    for j = 1:20
        Matrix20(i, j) = Input_Matrix(j+(i-1)*20); %���
    end
end
