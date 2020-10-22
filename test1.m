%二进制转十六进制
function [r7,r8] = test1(HI,HM)
leng7 = length(HI)/8;
n7 = mat2cell(HI,1,8*ones(1,leng7));
o7 = zeros(1,leng7);
for m7=1:leng7   
    for p7=1:8
        q7(p7) = num2str(n7{:,m7}(:,p7));
    end
    o7(:,m7) = bin2dec(q7);
end
r7 = char(o7);
% r7 = o7;

leng8 = length(HM)/8;
n8 = mat2cell(HM,1,8*ones(1,leng8));
o8 = zeros(1,leng8);

for m8=1:leng8   
    for p8=1:8
        q8(p8) = num2str(n8{:,m8}(:,p8));
    end
    o8(:,m8) = bin2dec(q8);
end
r8 = char(o8);
% r8 = o8;