src=imread('s.png');
dst=imread('dst.png');

%���ݼ�׼��  ������ֱ
src_in=imresize(src,[50,60]); %ѵ��������������
table_dst=imresize(dst,[50,60]);% ѵ�����������ݵı�ǩ
%test_src=imresize(c,[50,60]);% ���Լ�������
[m,n,k]=size(src_in);

p_src_in=[m*n,3];
t_table_dst=[m*n,3];
%p_test_c=[m*n,3];
result_show=zeros(m,n,3);
%������ֱ R G B
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
% for i=1:m
%     for j=1:n
%         p_test_c(count,1)= c(i,j,1); %R
%         p_test_c(count,2)= c(i,j,2);
%         p_test_c(count,3)= c(i,j,3);
%         count=count+1;
%     end
% end





%PΪ���� TΪ��ǩ
P=p_src_in;
T=t_table_dst;

[p1,minp,maxp,t1,mint,maxt]=premnmx(P,T);%��һ��
%��������
%net=newff(minmax(P),[3,12,3],{'tansig','tansig','purelin'},'trainlm');
net=newff(P',T',[3,12,10,10,10],{'tansig','tansig','tansig','tansig','purelin'},'trainlm');
%����ѵ������
net.trainParam.epochs = 200;
net.divideFcn = '';

%�����������
net.trainParam.goal=0.0001;
%ѵ������
[net,tr]=train(net,p1',t1');

%��������
a=p_src_in;
%���������ݹ�һ��
a=premnmx(a);
%���뵽�����������
b=sim(net,a');
%���õ������ݷ���һ���õ�Ԥ������
c=postmnmx(b',mint,maxt);

count=1;
for i=1:m
    for j=1:n
        result_show(i,j,1)= uint8(c(count,1)); %R
        result_show(i,j,2)= uint8(c(count,2));
        result_show(i,j,3)= uint8(c(count,3));
        count=count+1;
    end
end
figure;
imshow(result_show/256);
imwrite(result_show/256,'rescnn.jpg');
A = result_show/256;
save('file1.mat','A');   %������A���浽��ǰ�ļ����е��ļ�

Loss = 0.0;
for i=1:50
    for j=1:60
        Loss = Loss + (result_show(i,j,1)/256 - table_dst(i,j,1)/256)^2;
        Loss = Loss + (result_show(i,j,2)/256  - table_dst(i,j,2)/256 )^2;
        Loss = Loss + (result_show(i,j,3)/256  - table_dst(i,j,3)/256 )^2
        count=count+1;
    end
end
Loss

