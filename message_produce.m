% clear all;close all;clc;
% tic
%生成原始消息以及修改原始消息，P1包含多个数据子块
function [p1,MM] = message_produce()
    % a = rgb2gray((imread('lena.jpg')));
    % a = randi([0 255],32);      %生成不同数据量的原始消息
    % a = randi([0 255],112);
    % a = randi([0 255],352);
    % D = load('A1','a');
    % D = load('A2','a');
    D = load('A3','a');

    %改变一个比特
    [x5,y5] = size(D.a);
    MM = D.a;t5 = randi(x5);v5 = randi(y5);
    value = MM(t5,v5);
    va = dec2bin(value,8);  %十进制转成二进制，为字符串类型
    leng = length(va);
    z5 = zeros(1,leng);
    for i5=1:leng
        z5(i5) = str2double(va(i5));
    end
    i5 = randi(leng);
    if z5(i5)==1
        z5(i5) = 0;
    else
        z5(i5) = 1;
    end
    va1 = int2str(z5);
    va1(isspace(va1)) = [];  %去除字符串中的空格
    value1 = bin2dec(va1);
    MM(t5,v5) = value1;

    %将数据排成行数组，方便将消息长度添加在后
    b7 = dec2bin(D.a,8);  %原始消息
    c7 = double(b7);   %转成ascii码
    b8 = dec2bin(MM,8);  %修改后的消息
    c8 = double(b8);
    d7 = zeros(size(c8));
    d8 = d7;
    for i=1:8
        d7(:,i) = str2num(char(c7(:,i))) ;
        d8(:,i) = str2num(char(c8(:,i))) ;
    end
    e7 = d7';f7 = e7(:);g7 = f7';  %排成1行
    e8 = d8';f8 = e8(:);g8 = f8';  %排成1行

    %把消息长度放后面
    leng1 = length(g7);
    bin = dec2bin(leng1,64);
    l = zeros(1,64);
    for k=1:length(bin)
        l(k) = str2double(bin(k));
    end
    h =ones(1,2048);
    for i=1985:2048
        h(:,i) = l(:,i-1984);
    end

    j7 = cat(2,g7,h); %等同于[g7, h]
    j8 = cat(2,g8,h);
    q7 = int2str(j7);
    q8 = int2str(j8);
    q7(isspace(q7)) = [];
    q8(isspace(q8)) = [];

    %多重速率填充:在消息末尾填充100...001
    q7_copy = zeros(1, length(q7));
    q8_copy = zeros(1, length(q8));
    for i = 1:length(q7)
        q7_copy(i) = str2double(q7(i));
        q8_copy(i) = str2double(q8(i));
    end
    len_add = mod(length(q7_copy), 2048);
    data_supplyment_num = 2048 - len_add;
    data_supplyment = zeros(1, data_supplyment_num);
    data_supplyment(1) = 1;
    data_supplyment(data_supplyment_num) = 1;
    j7 = cat(2,q7_copy,data_supplyment); %等同于[g7, h]
    j8 = cat(2,q8_copy,data_supplyment);
    q7 = int2str(j7);
    q8 = int2str(j8);
    q7(isspace(q7)) = [];
    q8(isspace(q8)) = [];
    
    for i = 1:length(q7)
        M7(1,i) = str2double(q7(1,i));
        M8(1,i) = str2double(q8(1,i));
    end
    
    leng2 = length(j7)/2048;
    p1 = mat2cell(j7,1,2048*ones(1,leng2));
    MM = mat2cell(j8,1,2048*ones(1,leng2));  %每个元组包含256位二进制数据，用p1{：，1}形式进行索引
    
%     for i=1:leng2
%         o7(i) = bin2dec(n7{:,i});
%         o8(i) = bin2dec(n8{:,i});
%     end

%     %暂不填充至20×20
%     s = length(o7)/256;
%     p1 = mat2cell(o7,1,256*ones(1,s)); 
%     MM = mat2cell(o8,1,256*ones(1,s)); %转化成1×256的元胞数组，为后续转成16×16做准备
%     for t=1:s
%         p1{:,t} = reshape(p1{:,t},16,16);  %输出原始数据
%         MM{:,t} = reshape(MM{:,t},16,16);  %修改后的数据
%     end

% toc