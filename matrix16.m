%clc;clear all;
function Matrix16 = matrix16(Input_Matrix)
%%%�������2048λ��������ת��Ϊ[0,255]��ʮ�����������������г�16��16�ľ���

%%%��ʼ��
M_Matrix = zeros(size(2048/8, 8));
Str_Matrix = char(zeros(size(2048/8, 8)));
Matrix16 = zeros(16);

for i = 1:2048/8
    for j = 1:8
        M_Matrix(i, j) = Input_Matrix(j+(i-1)*8); %Input_MatrixΪ����Ķ��������ݣ�M_MatrixΪ�м�״̬����
    end
end
M_Matrix = num2str(M_Matrix);
M_Matrix(isspace(M_Matrix)) = [];  %ת��Ϊ�ַ���
for i = 1:2048/8
    for j = 1:8
        Str_Matrix(i, j) = M_Matrix(j+(i-1)*8); %�������г�256��8����ʽ
    end
end
K = bin2dec(Str_Matrix); %ת��Ϊʮ����

for i = 1:16
    for j = 1:16
        Matrix16(i, j) = K(j+(i-1)*16); %���
    end
end


