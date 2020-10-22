function P = xor1(M, N)
%%%�����������󣬽���ת��Ϊ�����������������ֵ
% M = intial_value();
% M = M(:); M = M';
% N = randi([0, 255], 16);

M = double(dec2bin(M, 8));
N = double(dec2bin(N, 8));  %ת��ASCII��

M_ = zeros(size(M));
N_ = zeros(size(N));

for i=1:8
    M_(:,i) = str2num(char(M(:,i))) ;
    N_(:,i) = str2num(char(N(:,i))) ;
end

M_ = M_'; M_ = M_(:); M_ = M_';
N_ = N_'; N_ = N_(:); N_ = N_';

P = xor(M, N);

