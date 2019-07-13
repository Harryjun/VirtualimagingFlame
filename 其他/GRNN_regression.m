clc
clear
% a=imread('p.png');
% b=imread('t.png');
% c=imread('pp.png');
a=imread('s.png');
b=imread('dst.png');
c=imread('s.png');

figure;
subplot(2,2,1);
imshow(a);

subplot(2,2,2);
imshow(b);

subplot(2,2,3);
imshow(c);

[m,n,k]=size(b);
a=imresize(a,[m,n]);
b=imresize(b,[m,n]);
c=imresize(c,[m,n]);

p_a=[m*n,3];
t_b=[m*n,3];
p_test_c=[m*n,3];
im_show=zeros(m,n,3);
%按行拉直 R G B
count=1;
for i=1:m
    for j=1:n
        p_a(count,1)= a(i,j,1); %R
        p_a(count,2)= a(i,j,2);
        p_a(count,3)= a(i,j,3);
        count=count+1;
    end
end

count=1;
for i=1:m
    for j=1:n
        t_b(count,1)= b(i,j,1); %R
        t_b(count,2)= b(i,j,2);
        t_b(count,3)= b(i,j,3);
        count=count+1;
    end
end

count=1;
for i=1:m
    for j=1:n
        p_test_c(count,1)= c(i,j,1); %R
        p_test_c(count,2)= c(i,j,2);
        p_test_c(count,3)= c(i,j,3);
        count=count+1;
    end
end


%v=zeros(3,3000);
nn=newgrnn(p_a',t_b');
v=sim(nn,p_test_c');




count=1;
for i=1:m
    for j=1:n
        im_show(i,j,1)= uint8(v(1,count)); %R
        im_show(i,j,2)= uint8(v(2,count));
        im_show(i,j,3)= uint8(v(3,count));
        count=count+1;
    end
end
subplot(2,2,4);
imshow(im_show/256);
imwrite(im_show/256,'res.jpg');
im_show=imresize(im_show,[50,60]);

B = im_show/256;
save('file2.mat','B');   %将变量A保存到当前文件夹中的文件


%滤波部分
 r = im_show(:,:,1);
 g = im_show(:,:,2);
 b = im_show(:,:,3);
 r = medfilt2(r,[3 3]);
 g = medfilt2(g,[3 3]);
 b = medfilt2(b,[3 3]);
 K1= cat(3,r,g,b); %对彩色图像R,G，B三个通道分别进行3×3模板的中值滤波 cat函数用于连接两个矩阵或数组
figure;
imshow(K1/256);


%计算错误率
accurate=zeros(1,4);%R，G, B MAP
temp=K1;
%temp=im_show;
count_accurate=0;
accurate_adjust=0.2;
percent=zeros(3,3000);
percent_abs=zeros(3,3000);
count_per=0;
d=imresize(a,[50,60]);
Loss = 0.0;
for i=1:50%欧式距离平方
    for j=1:60
        Loss = Loss + (im_show(i,j,1)/256 - d(i,j,1)/256)^2;
        Loss = Loss + (im_show(i,j,2)/256 - d(i,j,2)/256)^2;
        Loss = Loss + (im_show(i,j,3)/256 - d(i,j,3)/256)^2;
        count=count+1;
    end
end
Loss
for k=1:3
    for i=1:m
        for j=1:n
            count_per=count_per+1;
            percent(k,count_per)=(double(c(i,j,k))-temp(i,j,k))/double(c(i,j,k));
            percent_abs(k,count_per)=abs((double(c(i,j,k))-temp(i,j,k))/double(c(i,j,k)));
            if(percent_abs(k,count_per)<accurate_adjust)
                count_accurate=count_accurate+1;
            end
        end
    end
    accurate(k)=count_accurate/(m*n);
    count_accurate=0;
    count_per=0;
end
accurate(4)=sum(accurate(:))/3;

%绘制误差曲线
cc=0;
for i=1:3
    for j=1:3000
        if(abs(percent(i,j))>0.9)
            percent(i,j)=0;
            cc=cc+1;
        end
    end
end
figure;
x1=1:3000;
x2=1:3000;
x3=1:3000;
plot(x1,percent(1,:),'r-',x2,percent(2,:),'g-',x3,percent(3,:),'b-');


