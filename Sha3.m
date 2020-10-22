%����һ������ȥ�㻯��sobel����
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
D = load('A','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10');   %10�������λ��Ĥ��
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

    %���ս׶�
    for t1=1:486
%         p1{:,t1} = cat(2,p1{:,t1},zeros(1,144));
%         MM{:,t1} = cat(2,MM{:,t1},zeros(1,144));
            p = xor(p1{:,t1}, p4);
            M = xor(MM{:,t1}, p5);
            matrix_p = matrix16(p);
            matrix_M = matrix16(M);

    %��������ɢ�����
            lamda = 6328*10^(-10);k = 2*pi/lamda;
            L = 2.65*10^(-3);                            %ͼ��Ŀ�ߵ�λ��
            d1 = 0.25;d2 = 1*10^(-3);d3 = 0.25;

            u = linspace(-1./2./L,1./2./L,16).*16;
            v = linspace(-1./2./L,1./2./L,16).*16;
            [u,v] = meshgrid(u,v);

            H1 = exp(1i*k*d1.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H1���ݺ���
            fp = fftshift(fft2(matrix_p));
            ffp = fp.*H1;
            U1 = ifft2(ffp);%m1ǰ�ⳡ
            UM1 = U1.*M1;
            %%%%%%
            fp1 = fftshift(fft2(matrix_M));
            ffp1 = fp1.*H1;
            U11 = ifft2(ffp1);%m1ǰ�ⳡ
            UM11 = U11.*M1;

            H2 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H2���ݺ���
            fUM1 = fftshift(fft2(UM1));
            ffUM1 = fUM1.*H2;
            U2 = ifft2(ffUM1);%m2ǰ�ⳡ
            UM2 = U2.*M2;
            %%%%%
            fUM11 = fftshift(fft2(UM11));
            ffUM11 = fUM11.*H2;
            U21 = ifft2(ffUM11);%m2ǰ�ⳡ
            UM21 = U21.*M2;

            H3 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H3���ݺ���
            fUM2 = fftshift(fft2(UM2));
            ffUM2 = fUM2.*H3;
            U3 = ifft2(ffUM2);%m3ǰ�ⳡ
            UM3 = U3.*M3;
            %%%%
            fUM21 = fftshift(fft2(UM21));
            ffUM21 = fUM21.*H3;
            U31 = ifft2(ffUM21);%m3ǰ�ⳡ
            UM31 = U31.*M3;

            H4 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H4���ݺ���
            fUM3 = fftshift(fft2(UM3));
            ffUM3 = fUM3.*H4;
            U4 = ifft2(ffUM3);%M4ǰ�ⳡ
            UM4 = U4.*M4;
            %%%%
            fUM31 = fftshift(fft2(UM31));
            ffUM31 = fUM31.*H4;
            U41 = ifft2(ffUM31);%m4ǰ�ⳡ
            UM41 = U41.*M4;

            H5 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H5���ݺ���
            fUM4 = fftshift(fft2(UM4));
            ffUM4 = fUM4.*H5;
            U5 = ifft2(ffUM4);%M5ǰ�ⳡ
            UM5 = U5.*M5;
            %%%%
            fUM41 = fftshift(fft2(UM41));
            ffUM41 = fUM41.*H5;
            U51 = ifft2(ffUM41);%m5ǰ�ⳡ
            UM51 = U51.*M5;

            H6 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H6���ݺ���
            fUM5 = fftshift(fft2(UM5));
            ffUM5 = fUM5.*H6;
    %       I = abs(ifft2(ffUM5));%M6ǰ�ⳡ
            U6 = ifft2(ffUM5);
            UM6 = U6.*M6;
            %%%%
            fUM51 = fftshift(fft2(UM51));
            ffUM51 = fUM51.*H6;
    %       I1 = abs(ifft2(ffUM51));%m6ǰ�ⳡ
            U61 = ifft2(ffUM51);
            UM61 = U61.*M6;

            H7 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H7���ݺ���
            fUM6 = fftshift(fft2(UM6));
            ffUM6 = fUM6.*H7;
    %         I = abs(ifft2(ffUM6));
            U7 = ifft2(ffUM6);%M7ǰ�ⳡ
            UM7 = U7.*M7;
             %%%%
            fUM61 = fftshift(fft2(UM61));
            ffUM61 = fUM61.*H7;
    %         I1 = abs(ifft2(ffUM61));
            U71 = ifft2(ffUM61);%m7ǰ�ⳡ
            UM71 = U71.*M7;

            H8 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H8���ݺ���
            fUM7 = fftshift(fft2(UM7));
            ffUM7 = fUM7.*H8;
            U8 = ifft2(ffUM7);%M7ǰ�ⳡ
            UM8 = U8.*M8;
            %%%%
            fUM71 = fftshift(fft2(UM71));
            ffUM71 = fUM71.*H8;
            U81 = ifft2(ffUM71);%m7ǰ�ⳡ
            UM81 = U81.*M8;

            H9 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H9���ݺ���
            fUM8 = fftshift(fft2(UM8));
            ffUM8 = fUM8.*H9;
            U9 = ifft2(ffUM8);%M9ǰ�ⳡ
            UM9 = U9.*M9;
             %%%%
            fUM81 = fftshift(fft2(UM81));
            ffUM81 = fUM81.*H9;
            U91 = ifft2(ffUM81);%m9ǰ�ⳡ
            UM91 = U91.*M9;

            H10 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H10���ݺ���
            fUM9 = fftshift(fft2(UM9));
            ffUM9 = fUM9.*H10;
            U10 = ifft2(ffUM9);%M10ǰ�ⳡ
            UM10 = U10.*M10;
            %%%%
            fUM91 = fftshift(fft2(UM91));
            ffUM91 = fUM91.*H10;
            U101 = ifft2(ffUM91);%m10ǰ�ⳡ
            UM101 = U101.*M10;

            H11 = exp(1i*k*d3.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H11���ݺ���
            fUM10 = fftshift(fft2(UM10));
            ffUM10 = fUM10.*H11;
            I = abs(ifft2(ffUM10));  %ԭʼ��Ϣ��Ӧ��ɢ��
            %figure,imshow(I,[]),title('ɢ��ͼ0');
            %%%%
            fUM101 = fftshift(fft2(UM101));
            ffUM101 = fUM101.*H11;
            I1 = abs(ifft2(ffUM101));  %�޸ĺ����Ϣ��Ӧ��ɢ��
            %figure,imshow(I1,[]),title('ɢ��ͼ1');
            I = 255*(I-min(min(I)))/(max(max(I))-min(min(I)));
            I1 = 255*(I1-min(min(I1)))/(max(max(I1))-min(min(I1)));

            p4 = row2048(I);
            p5 = row2048(I1);

    end
    for i = 1:128          %ǰ128λhashֵ
        Hash_p(1,i) = p4(1,i);
        Hash_M(1,i) = p5(1,i);
    end
    %��ѹ�׶�
    for i = 1:3
            matrix_p = matrix16(p4);
            matrix_M = matrix16(p5);
            lamda = 6328*10^(-10);k = 2*pi/lamda;
            L = 2.65*10^(-3);                            %ͼ��Ŀ�ߵ�λ��
            d1 = 0.25;d2 = 1*10^(-3);d3 = 0.25;

            u = linspace(-1./2./L,1./2./L,16).*16;
            v = linspace(-1./2./L,1./2./L,16).*16;
            [u,v] = meshgrid(u,v);

            H1 = exp(1i*k*d1.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H1���ݺ���
            fp = fftshift(fft2(matrix_p));
            ffp = fp.*H1;
            U1 = ifft2(ffp);%m1ǰ�ⳡ
            UM1 = U1.*M1;
            %%%%%%
            fp1 = fftshift(fft2(matrix_M));
            ffp1 = fp1.*H1;
            U11 = ifft2(ffp1);%m1ǰ�ⳡ
            UM11 = U11.*M1;

            H2 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H2���ݺ���
            fUM1 = fftshift(fft2(UM1));
            ffUM1 = fUM1.*H2;
            U2 = ifft2(ffUM1);%m2ǰ�ⳡ
            UM2 = U2.*M2;
            %%%%%
            fUM11 = fftshift(fft2(UM11));
            ffUM11 = fUM11.*H2;
            U21 = ifft2(ffUM11);%m2ǰ�ⳡ
            UM21 = U21.*M2;

            H3 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H3���ݺ���
            fUM2 = fftshift(fft2(UM2));
            ffUM2 = fUM2.*H3;
            U3 = ifft2(ffUM2);%m3ǰ�ⳡ
            UM3 = U3.*M3;
            %%%%
            fUM21 = fftshift(fft2(UM21));
            ffUM21 = fUM21.*H3;
            U31 = ifft2(ffUM21);%m3ǰ�ⳡ
            UM31 = U31.*M3;

            H4 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H4���ݺ���
            fUM3 = fftshift(fft2(UM3));
            ffUM3 = fUM3.*H4;
            U4 = ifft2(ffUM3);%M4ǰ�ⳡ
            UM4 = U4.*M4;
            %%%%
            fUM31 = fftshift(fft2(UM31));
            ffUM31 = fUM31.*H4;
            U41 = ifft2(ffUM31);%m4ǰ�ⳡ
            UM41 = U41.*M4;

            H5 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H5���ݺ���
            fUM4 = fftshift(fft2(UM4));
            ffUM4 = fUM4.*H5;
            U5 = ifft2(ffUM4);%M5ǰ�ⳡ
            UM5 = U5.*M5;
            %%%%
            fUM41 = fftshift(fft2(UM41));
            ffUM41 = fUM41.*H5;
            U51 = ifft2(ffUM41);%m5ǰ�ⳡ
            UM51 = U51.*M5;

            H6 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H6���ݺ���
            fUM5 = fftshift(fft2(UM5));
            ffUM5 = fUM5.*H6;
    %       I = abs(ifft2(ffUM5));%M6ǰ�ⳡ
            U6 = ifft2(ffUM5);
            UM6 = U6.*M6;
            %%%%
            fUM51 = fftshift(fft2(UM51));
            ffUM51 = fUM51.*H6;
    %       I1 = abs(ifft2(ffUM51));%m6ǰ�ⳡ
            U61 = ifft2(ffUM51);
            UM61 = U61.*M6;

            H7 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H7���ݺ���
            fUM6 = fftshift(fft2(UM6));
            ffUM6 = fUM6.*H7;
    %         I = abs(ifft2(ffUM6));
            U7 = ifft2(ffUM6);%M7ǰ�ⳡ
            UM7 = U7.*M7;
             %%%%
            fUM61 = fftshift(fft2(UM61));
            ffUM61 = fUM61.*H7;
    %         I1 = abs(ifft2(ffUM61));
            U71 = ifft2(ffUM61);%m7ǰ�ⳡ
            UM71 = U71.*M7;

            H8 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H8���ݺ���
            fUM7 = fftshift(fft2(UM7));
            ffUM7 = fUM7.*H8;
            U8 = ifft2(ffUM7);%M7ǰ�ⳡ
            UM8 = U8.*M8;
            %%%%
            fUM71 = fftshift(fft2(UM71));
            ffUM71 = fUM71.*H8;
            U81 = ifft2(ffUM71);%m7ǰ�ⳡ
            UM81 = U81.*M8;

            H9 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H9���ݺ���
            fUM8 = fftshift(fft2(UM8));
            ffUM8 = fUM8.*H9;
            U9 = ifft2(ffUM8);%M9ǰ�ⳡ
            UM9 = U9.*M9;
             %%%%
            fUM81 = fftshift(fft2(UM81));
            ffUM81 = fUM81.*H9;
            U91 = ifft2(ffUM81);%m9ǰ�ⳡ
            UM91 = U91.*M9;

            H10 = exp(1i*k*d2.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H10���ݺ���
            fUM9 = fftshift(fft2(UM9));
            ffUM9 = fUM9.*H10;
            U10 = ifft2(ffUM9);%M10ǰ�ⳡ
            UM10 = U10.*M10;
            %%%%
            fUM91 = fftshift(fft2(UM91));
            ffUM91 = fUM91.*H10;
            U101 = ifft2(ffUM91);%m10ǰ�ⳡ
            UM101 = U101.*M10;

            H11 = exp(1i*k*d3.*(1-lamda.*lamda.*(u.*u+v.*v)./2));%H11���ݺ���
            fUM10 = fftshift(fft2(UM10));
            ffUM10 = fUM10.*H11;
            I = abs(ifft2(ffUM10));  %ԭʼ��Ϣ��Ӧ��ɢ��
            %figure,imshow(I,[]),title('ɢ��ͼ0');
            %%%%
            fUM101 = fftshift(fft2(UM101));
            ffUM101 = fUM101.*H11;
            I1 = abs(ifft2(ffUM101));  %�޸ĺ����Ϣ��Ӧ��ɢ��
            %figure,imshow(I1,[]),title('ɢ��ͼ1');
            I = 255*(I-min(min(I)))/(max(max(I))-min(min(I)));
            I1 = 255*(I1-min(min(I1)))/(max(max(I1))-min(min(I1)));

            p4 = row2048(I);
            p5 = row2048(I1);
            
            for j = (128*i+1):(128*i+128)  %��384λ
                Hash_p(1,j) = p4(1,j);
                Hash_M(1,j) = p5(1,j);
            end

    end
    
%     p4;
%     p5; 
    %����ԭʼ��Ϣ�޸ĺ�hashֵ�仯��λ��
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

%     %����ײ��
%     [r7,r8] = test1(HI,HM);%תASCLL���ַ�
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
B1 = sum(sum(count))/K;  %����λ����ƽ��ֵ
AVE = sum(sum(E1))/K;
E = B1/total;
B2 = 0;
B4 = 0;
for kk=1:K
    B2 = B2 + (count(:,kk)-B1)^2;
    B4 = B4 + (E1(:,kk)-AVE)^2;
end
B3 = sqrt(B2/(K-1));     %����λ������׼��  
B5 = sqrt(B4/(K-1));  

toc

 