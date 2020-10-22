%不归一化，不去零化，sobel算子
clear all;close all;clc;

tic
K = 10000;
count = zeros(1,K);
E1 = zeros(1,K);
shot1 = zeros(1,K);
CC2 = zeros(1,K);
shot2 = 0;
Oneshot = 0;
Twoshot = 0;
Hash_p = zeros(1,512);
Hash_M = zeros(1,512);
% r1 = rand(16,16);r2 = rand(16,16);r3 = rand(16,16);r4 = rand(16,16);r5 = rand(16,16);
% r6 = rand(16,16);r7 = rand(16,16);r8 = rand(16,16);r9 = rand(16,16);r10 = rand(16,16);
% save('A.mat','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10')
D = load('A','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10');   %10层随机相位掩膜板
M1 = exp(1i*2*pi.*D.r1);M2 = exp(1i*2*pi.*D.r2);M3 = exp(1i*2*pi.*D.r3);
M4 = exp(1i*2*pi.*D.r4);M5 = exp(1i*2*pi.*D.r5);M6 = exp(1i*2*pi.*D.r6);
M7 = exp(1i*2*pi.*D.r7);M8 = exp(1i*2*pi.*D.r8);M9 = exp(1i*2*pi.*D.r9);
M10 = exp(1i*2*pi.*D.r10);

%-----------------------------------------------------------

R = zeros(1,K);
X = 1:K;
figure('color',[1 1 1]),h = plot(X(1), R(1), 'EraseMode', 'none');
axis([1 K 0 1]);
xlabel('test time N')
ylabel('avalanche effect coefficient')
%-------------------------------------------------------------

for ii=1:K
    p4 = initial_value();%
    p5 = initial_value();%

    [p1,MM] = message_produce();
    [x,y] = size(p1{1,1});

    %吸收阶段
    for t1=1:486
%         p1{:,t1} = cat(2,p1{:,t1},zeros(1,144));
%         MM{:,t1} = cat(2,MM{:,t1},zeros(1,144));
            p = xor(p1{:,t1}, p4);
            M = xor(MM{:,t1}, p5);
            matrix_p = matrix16(p);
            matrix_M = matrix16(M);

    %经过多重散射介质
            lamda = 6328*10^(-10);k = 2*pi/lamda;
            L = 2.65*10^(-3);                            %图像的宽高单位米
            d1 = 0.25;d2 = 1*10^(-3);d3 = 0.25;

            u = linspace(-1./2./L,1./2./L,16).*16;
            v = linspace(-1./2./L,1./2./L,16).*16;
            [u,v] = meshgrid(u,v);

            H1 = exp(1i*k*d1.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H1传递函数
            fp = fftshift(fft2(matrix_p));
            ffp = fp.*H1;
            U1 = ifft2(ffp);%m1前光场
            UM1 = U1.*M1;
            %%%%%%
            fp1 = fftshift(fft2(matrix_M));
            ffp1 = fp1.*H1;
            U11 = ifft2(ffp1);%m1前光场
            UM11 = U11.*M1;

            H2 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H2传递函数
            fUM1 = fftshift(fft2(UM1));
            ffUM1 = fUM1.*H2;
            U2 = ifft2(ffUM1);%m2前光场
            UM2 = U2.*M2;
            %%%%%
            fUM11 = fftshift(fft2(UM11));
            ffUM11 = fUM11.*H2;
            U21 = ifft2(ffUM11);%m2前光场
            UM21 = U21.*M2;

            H3 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H3传递函数
            fUM2 = fftshift(fft2(UM2));
            ffUM2 = fUM2.*H3;
            U3 = ifft2(ffUM2);%m3前光场
            UM3 = U3.*M3;
            %%%%
            fUM21 = fftshift(fft2(UM21));
            ffUM21 = fUM21.*H3;
            U31 = ifft2(ffUM21);%m3前光场
            UM31 = U31.*M3;

            H4 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H4传递函数
            fUM3 = fftshift(fft2(UM3));
            ffUM3 = fUM3.*H4;
            U4 = ifft2(ffUM3);%M4前光场
            UM4 = U4.*M4;
            %%%%
            fUM31 = fftshift(fft2(UM31));
            ffUM31 = fUM31.*H4;
            U41 = ifft2(ffUM31);%m4前光场
            UM41 = U41.*M4;

            H5 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H5传递函数
            fUM4 = fftshift(fft2(UM4));
            ffUM4 = fUM4.*H5;
            U5 = ifft2(ffUM4);%M5前光场
            UM5 = U5.*M5;
            %%%%
            fUM41 = fftshift(fft2(UM41));
            ffUM41 = fUM41.*H5;
            U51 = ifft2(ffUM41);%m5前光场
            UM51 = U51.*M5;

            H6 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H6传递函数
            fUM5 = fftshift(fft2(UM5));
            ffUM5 = fUM5.*H6;
    %       I = abs(ifft2(ffUM5));%M6前光场
            U6 = ifft2(ffUM5);
            UM6 = U6.*M6;
            %%%%
            fUM51 = fftshift(fft2(UM51));
            ffUM51 = fUM51.*H6;
    %       I1 = abs(ifft2(ffUM51));%m6前光场
            U61 = ifft2(ffUM51);
            UM61 = U61.*M6;

            H7 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H7传递函数
            fUM6 = fftshift(fft2(UM6));
            ffUM6 = fUM6.*H7;
    %         I = abs(ifft2(ffUM6));
            U7 = ifft2(ffUM6);%M7前光场
            UM7 = U7.*M7;
             %%%%
            fUM61 = fftshift(fft2(UM61));
            ffUM61 = fUM61.*H7;
    %         I1 = abs(ifft2(ffUM61));
            U71 = ifft2(ffUM61);%m7前光场
            UM71 = U71.*M7;

            H8 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H8传递函数
            fUM7 = fftshift(fft2(UM7));
            ffUM7 = fUM7.*H8;
            U8 = ifft2(ffUM7);%M7前光场
            UM8 = U8.*M8;
            %%%%
            fUM71 = fftshift(fft2(UM71));
            ffUM71 = fUM71.*H8;
            U81 = ifft2(ffUM71);%m7前光场
            UM81 = U81.*M8;

            H9 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H9传递函数
            fUM8 = fftshift(fft2(UM8));
            ffUM8 = fUM8.*H9;
            U9 = ifft2(ffUM8);%M9前光场
            UM9 = U9.*M9;
             %%%%
            fUM81 = fftshift(fft2(UM81));
            ffUM81 = fUM81.*H9;
            U91 = ifft2(ffUM81);%m9前光场
            UM91 = U91.*M9;

            H10 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H10传递函数
            fUM9 = fftshift(fft2(UM9));
            ffUM9 = fUM9.*H10;
            U10 = ifft2(ffUM9);%M10前光场
            UM10 = U10.*M10;
            %%%%
            fUM91 = fftshift(fft2(UM91));
            ffUM91 = fUM91.*H10;
            U101 = ifft2(ffUM91);%m10前光场
            UM101 = U101.*M10;

            H11 = exp(1i*k*d3.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H11传递函数
            fUM10 = fftshift(fft2(UM10));
            ffUM10 = fUM10.*H11;
            I = abs(ifft2(ffUM10));  %原始消息对应的散斑
            %figure,imshow(I,[]),title('散斑图0');
            %%%%
            fUM101 = fftshift(fft2(UM101));
            ffUM101 = fUM101.*H11;
            I1 = abs(ifft2(ffUM101));  %修改后的消息对应的散斑
            %figure,imshow(I1,[]),title('散斑图1');
            I = 255*(I-min(min(I)))/(max(max(I))-min(min(I)));
            I1 = 255*(I1-min(min(I1)))/(max(max(I1))-min(min(I1)));

            p4 = row2048(I);
            p5 = row2048(I1);

    end
    for i = 1:128          %前128位hash值
        Hash_p(1,i) = p4(1,i);
        Hash_M(1,i) = p5(1,i);
    end
    %挤压阶段
    for i = 1:3
            matrix_p = matrix16(p4);
            matrix_M = matrix16(p5);
            lamda = 6328*10^(-10);k = 2*pi/lamda;
            L = 2.65*10^(-3);                            %图像的宽高单位米
            d1 = 0.25;d2 = 1*10^(-3);d3 = 0.25;

            u = linspace(-1./2./L,1./2./L,16).*16;
            v = linspace(-1./2./L,1./2./L,16).*16;
            [u,v] = meshgrid(u,v);

            H1 = exp(1i*k*d1.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H1传递函数
            fp = fftshift(fft2(matrix_p));
            ffp = fp.*H1;
            U1 = ifft2(ffp);%m1前光场
            UM1 = U1.*M1;
            %%%%%%
            fp1 = fftshift(fft2(matrix_M));
            ffp1 = fp1.*H1;
            U11 = ifft2(ffp1);%m1前光场
            UM11 = U11.*M1;

            H2 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H2传递函数
            fUM1 = fftshift(fft2(UM1));
            ffUM1 = fUM1.*H2;
            U2 = ifft2(ffUM1);%m2前光场
            UM2 = U2.*M2;
            %%%%%
            fUM11 = fftshift(fft2(UM11));
            ffUM11 = fUM11.*H2;
            U21 = ifft2(ffUM11);%m2前光场
            UM21 = U21.*M2;

            H3 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H3传递函数
            fUM2 = fftshift(fft2(UM2));
            ffUM2 = fUM2.*H3;
            U3 = ifft2(ffUM2);%m3前光场
            UM3 = U3.*M3;
            %%%%
            fUM21 = fftshift(fft2(UM21));
            ffUM21 = fUM21.*H3;
            U31 = ifft2(ffUM21);%m3前光场
            UM31 = U31.*M3;

            H4 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H4传递函数
            fUM3 = fftshift(fft2(UM3));
            ffUM3 = fUM3.*H4;
            U4 = ifft2(ffUM3);%M4前光场
            UM4 = U4.*M4;
            %%%%
            fUM31 = fftshift(fft2(UM31));
            ffUM31 = fUM31.*H4;
            U41 = ifft2(ffUM31);%m4前光场
            UM41 = U41.*M4;

            H5 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H5传递函数
            fUM4 = fftshift(fft2(UM4));
            ffUM4 = fUM4.*H5;
            U5 = ifft2(ffUM4);%M5前光场
            UM5 = U5.*M5;
            %%%%
            fUM41 = fftshift(fft2(UM41));
            ffUM41 = fUM41.*H5;
            U51 = ifft2(ffUM41);%m5前光场
            UM51 = U51.*M5;

            H6 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H6传递函数
            fUM5 = fftshift(fft2(UM5));
            ffUM5 = fUM5.*H6;
    %       I = abs(ifft2(ffUM5));%M6前光场
            U6 = ifft2(ffUM5);
            UM6 = U6.*M6;
            %%%%
            fUM51 = fftshift(fft2(UM51));
            ffUM51 = fUM51.*H6;
    %       I1 = abs(ifft2(ffUM51));%m6前光场
            U61 = ifft2(ffUM51);
            UM61 = U61.*M6;

            H7 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H7传递函数
            fUM6 = fftshift(fft2(UM6));
            ffUM6 = fUM6.*H7;
    %         I = abs(ifft2(ffUM6));
            U7 = ifft2(ffUM6);%M7前光场
            UM7 = U7.*M7;
             %%%%
            fUM61 = fftshift(fft2(UM61));
            ffUM61 = fUM61.*H7;
    %         I1 = abs(ifft2(ffUM61));
            U71 = ifft2(ffUM61);%m7前光场
            UM71 = U71.*M7;

            H8 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H8传递函数
            fUM7 = fftshift(fft2(UM7));
            ffUM7 = fUM7.*H8;
            U8 = ifft2(ffUM7);%M7前光场
            UM8 = U8.*M8;
            %%%%
            fUM71 = fftshift(fft2(UM71));
            ffUM71 = fUM71.*H8;
            U81 = ifft2(ffUM71);%m7前光场
            UM81 = U81.*M8;

            H9 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H9传递函数
            fUM8 = fftshift(fft2(UM8));
            ffUM8 = fUM8.*H9;
            U9 = ifft2(ffUM8);%M9前光场
            UM9 = U9.*M9;
             %%%%
            fUM81 = fftshift(fft2(UM81));
            ffUM81 = fUM81.*H9;
            U91 = ifft2(ffUM81);%m9前光场
            UM91 = U91.*M9;

            H10 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H10传递函数
            fUM9 = fftshift(fft2(UM9));
            ffUM9 = fUM9.*H10;
            U10 = ifft2(ffUM9);%M10前光场
            UM10 = U10.*M10;
            %%%%
            fUM91 = fftshift(fft2(UM91));
            ffUM91 = fUM91.*H10;
            U101 = ifft2(ffUM91);%m10前光场
            UM101 = U101.*M10;

            H11 = exp(1i*k*d3.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H11传递函数
            fUM10 = fftshift(fft2(UM10));
            ffUM10 = fUM10.*H11;
            I = abs(ifft2(ffUM10));  %原始消息对应的散斑
            %figure,imshow(I,[]),title('散斑图0');
            %%%%
            fUM101 = fftshift(fft2(UM101));
            ffUM101 = fUM101.*H11;
            I1 = abs(ifft2(ffUM101));  %修改后的消息对应的散斑
            %figure,imshow(I1,[]),title('散斑图1');
            I = 255*(I-min(min(I)))/(max(max(I))-min(min(I)));
            I1 = 255*(I1-min(min(I1)))/(max(max(I1))-min(min(I1)));

            p4 = row2048(I);
            p5 = row2048(I1);
            
            for j = (128*i+1):(128*i+128)  %后384位
                Hash_p(1,j) = p4(1,j);
                Hash_M(1,j) = p5(1,j);
            end

    end
    
%     p4;
%     p5; 
    %计算原始消息修改后hash值变化的位数
    MI = xor(Hash_p,Hash_M);
    tot = length(Hash_p);
    for i=1:size(MI,2)
        if MI(:,i) == 1
            count(:,ii) = count(:,ii)+1;
        end
        E1(:,ii) = count(:,ii)/tot;
    end
    
%     p2 = sobel(I);
%     p3 = sobel(I1);
%     CC1 = corr2(p2,p3);
%     CC2(ii) = CC1;

    %-------------------------------------------------
    R(ii) = E1(:,ii);
    set(h,'XData',X(1:ii),'YData',R(1:ii));
    drawnow
    %-------------------------------------------------

%     %抗碰撞性
%     [r7,r8] = test1(HI,HM);%转ASCLL码字符
%     % r7 = [1 0 5 0 7 9];
%     % r8 = [1 7 3 8 7 2];
%     shot = 0;
%     for i=1:length(r7)
%         if r7(i) == r8(i)
%             shot = shot+1;
%         end
%     end
%     if shot == 1
%         Oneshot = Oneshot+1;
%     elseif shot == 2
%         Twoshot = Twoshot+1;
%     end
%     if shot ~= 0
%         shot2 = shot2+1;
%     end
%     shot1(ii) = shot;
end
  
total = length(HI);
B1 = sum(sum(count))/K;  %差异位数量平均值
AVE = sum(sum(E1))/K;
E = B1/total;
B2 = 0;
B4 = 0;
for kk=1:K
    B2 = B2 + (count(:,kk)-B1)^2;
    B4 = B4 + (E1(:,kk)-AVE)^2;
end
B3 = sqrt(B2/(K-1));     %差异位数量标准差  
B5 = sqrt(B4/(K-1));  

toc

 