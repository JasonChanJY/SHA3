% clear all;close all;clc;
% tic
%����ԭʼ��Ϣ�Լ��޸�ԭʼ��Ϣ��P1������������ӿ�
function [p1,MM] = message_produce()
    % a = rgb2gray((imread('lena.jpg')));
    % a = randi([0 255],32);      %���ɲ�ͬ��������ԭʼ��Ϣ
    % a = randi([0 255],112);
    % a = randi([0 255],352);
    % D = load('A1','a');
    % D = load('A2','a');
    D = load('A3','a');

    %�ı�һ������
    [x5,y5] = size(D.a);
    MM = D.a;t5 = randi(x5);v5 = randi(y5);
    value = MM(t5,v5);
    va = dec2bin(value,8);  %ʮ����ת�ɶ����ƣ�Ϊ�ַ�������
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
    va1(isspace(va1)) = [];  %ȥ���ַ����еĿո�
    value1 = bin2dec(va1);
    MM(t5,v5) = value1;

    %�������ų������飬���㽫��Ϣ��������ں�
    b7 = dec2bin(D.a,8);  %ԭʼ��Ϣ
    c7 = double(b7);   %ת��ascii��
    b8 = dec2bin(MM,8);  %�޸ĺ����Ϣ
    c8 = double(b8);
    d7 = zeros(size(c8));
    d8 = d7;
    for i=1:8
        d7(:,i) = str2num(char(c7(:,i))) ;
        d8(:,i) = str2num(char(c8(:,i))) ;
    end
    e7 = d7';f7 = e7(:);g7 = f7';  %�ų�1��
    e8 = d8';f8 = e8(:);g8 = f8';  %�ų�1��

    %����Ϣ���ȷź���
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

    j7 = cat(2,g7,h); %��ͬ��[g7, h]
    j8 = cat(2,g8,h);
    q7 = int2str(j7);
    q8 = int2str(j8);
    q7(isspace(q7)) = [];
    q8(isspace(q8)) = [];

    %�����������:����Ϣĩβ���100...001
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
    j7 = cat(2,q7_copy,data_supplyment); %��ͬ��[g7, h]
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
    MM = mat2cell(j8,1,2048*ones(1,leng2));  %ÿ��Ԫ�����256λ���������ݣ���p1{����1}��ʽ��������
    
%     for i=1:leng2
%         o7(i) = bin2dec(n7{:,i});
%         o8(i) = bin2dec(n8{:,i});
%     end

%     %�ݲ������20��20
%     s = length(o7)/256;
%     p1 = mat2cell(o7,1,256*ones(1,s)); 
%     MM = mat2cell(o8,1,256*ones(1,s)); %ת����1��256��Ԫ�����飬Ϊ����ת��16��16��׼��
%     for t=1:s
%         p1{:,t} = reshape(p1{:,t},16,16);  %���ԭʼ����
%         MM{:,t} = reshape(MM{:,t},16,16);  %�޸ĺ������
%     end

% toc