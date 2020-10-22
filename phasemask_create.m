% clear all;close all;clc;
% tic
function p2 = phasemask_create()
% rng(130754)
a = randi([0 1],[1 2048]);
b = mat2cell(a,1,8*ones(1,2048/8));
c = zeros(1,256);
for i=1:256
    for j=1:8
        d(j) = num2str(b{:,i}(:,j));
    end
    c(:,i) = bin2dec(d);
end
e = reshape(c,16,16);
% p2 = (e-min(min(e)))./(max(max(e))-min(min(e)));
p2 = e;
% toc
