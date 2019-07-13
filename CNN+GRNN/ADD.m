clc;
clear;
load('file1.mat')
load('file2.mat')

%CNN_feature = load('file1.mat','A')  % 加载 MAT 文件 filename 中的指定变量
%GRNN_feature = load('file2.mat','B')  % 加载 MAT 文件 filename 中的指定变量

count=1;
feature = [3000,6];
for i=1:50
    for j=1:60
        feature(count,1)= A(i,j,1); %R
        feature(count,2)= A(i,j,2);
        feature(count,3)= A(i,j,3);
        feature(count,4)= B(i,j,1); %R
        feature(count,5)= B(i,j,2);
        feature(count,6)= B(i,j,3);
        count=count+1;
    end
end


src=imread('s.png');
dst=imread('dst.png');




%数据集准备  按行拉直
src_in=imresize(src,[50,60]); %训练集的输入数据
table_dst=imresize(dst,[50,60]);% 训练集输入数据的标签
%test_src=imresize(c,[50,60]);% 测试集的输入
[m,n,k]=size(src_in);




p_src_in=[m*n,3];
t_table_dst=[m*n,3];
%p_test_c=[m*n,3];
result_show=zeros(m,n,3);
%按行拉直 R G B
count=1;
for i=1:m
    for j=1:n
        p_src_in(count,1)= src_in(i,j,1); %R
        p_src_in(count,2)= src_in(i,j,2);
        p_src_in(count,3)= src_in(i,j,3);
        count=count+1;
    end
end

count=1;
for i=1:m
    for j=1:n
        t_table_dst(count,1)= table_dst(i,j,1); %R
        t_table_dst(count,2)= table_dst(i,j,2);
        t_table_dst(count,3)= table_dst(i,j,3);
        count=count+1;
    end
end

count=1;

T=t_table_dst;

%[feature1,minp,maxp,t1,mint,maxt]=premnmx(feature,T);%归一化
%创建网络
%net=newff(minmax(P),[3,12,3],{'tansig','tansig','purelin'},'trainlm');
net=newff(feature',T',[3,10,20,10],{'tansig','tansig','tansig','purelin'},'trainlm');
%设置训练次数
net.trainParam.epochs = 2000;
net.divideFcn = '';

%设置收敛误差
net.trainParam.goal=0.0001;
%训练网络
[net,tr]=train(net,feature',T');

% % %输入数据
%  a=feature;
% % %将输入数据归一化
%  a=premnmx(a);
% % %放入到网络输出数据
b=sim(net,feature');
%将得到的数据反归一化得到预测数据
% c=postmnmx(b',mint,maxt);
c = b';
count=1;
result_show=zeros(50,60,3);
for i=1:50
    for j=1:60
        result_show(i,j,1)= uint8(c(count,1)); %R
        result_show(i,j,2)= uint8(c(count,2));
        result_show(i,j,3)= uint8(c(count,3));
        count=count+1;
    end
end
figure;
imshow(result_show/256);
%imwrite(result_show/256,'rescnn.jpg');
%A = result_show/256;

imwrite(result_show/256,'GRNN0-cnn.jpg');
Loss = 0;
for i=1:50
    for j=1:60
        Loss = Loss + (result_show(i,j,1) - table_dst(i,j,1))^2;
        Loss = Loss + (result_show(i,j,2) - table_dst(i,j,2))^2;
        Loss = Loss + (result_show(i,j,3) - table_dst(i,j,3))^2;
        count=count+1;
    end
end
Loss


