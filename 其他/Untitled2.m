clc
clear
a=imread('p.png');
b=imread('t.png');
c=imread('pp.png');

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
%������ֱ R G B
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

step=0.01:0.01:1;
v=zeros(3,3000,100);%�ڲ�ͬ��step��������ϳ���ͼ��


res_error=zeros(100,4);%R��G, B MAP
count=0;
for s=1:100
    nn=newgrnn(p_a',t_b',step(s));
    v(:,:,s)=sim(nn,p_test_c');
    count=count+1;
    for i=1:m
        for j=1:n
            im_show(i,j,1)= uint8(v(1,count,s)); %R
            im_show(i,j,2)= uint8(v(2,count,s));
            im_show(i,j,3)= uint8(v(3,count,s));
            count=count+1;
        end
    end
    count=0;
    temp=im_show;
    res_error=erro_caculate(c,temp,s,res_error);
end




%subplot(2,2,4);
%imshow(im_show/256);
%imwrite(im_show/256,'res.jpg');
%�˲�����
%  r = im_show(:,:,1);
%  g = im_show(:,:,2);
%  b = im_show(:,:,3);
%  r = medfilt2(r,[3 3]);
%  g = medfilt2(g,[3 3]);
%  b = medfilt2(b,[3 3]);
%  K1= cat(3,r,g,b); %�Բ�ɫͼ��R,G��B����ͨ���ֱ����3��3ģ�����ֵ�˲� cat�������������������������
% figure;
% imshow(K1/256);



%�����������
% x1=1:3000;
% x2=1:3000;
% x3=1:3000;
% plot(x1,percent(1,:),'r-',x2,percent(2,:),'g-',x3,percent(3,:),'b-');

%���������
function accurate =erro_caculate(target,pre_t,num,accurate)
        %accurate=zeros(100,4);%R��G, B MAP
        m=60;n=50;
        count_accurate=0;
        accurate_adjust=0.2;
        percent=zeros(3,3000);%������ ����ֵ����
        percent_abs=zeros(3,3000);%������� ֻ����ֵ
        count_per=0;
        for k=1:3
            for i=1:m
                for j=1:n
                    count_per=count_per+1;
                    percent(k,count_per)=(double(target(i,j,k))-pre_t(i,j,k))/double(target(i,j,k));
                    percent_abs(k,count_per)=abs((double(target(i,j,k))-pre_t(i,j,k))/double(target(i,j,k)));
                    if(percent_abs(k,count_per)<accurate_adjust)
                        count_accurate=count_accurate+1;
                    end
                end
            end
            accurate(num,k)=count_accurate/(m*n);
            count_accurate=0;
            count_per=0;
        end
        accurate(num,4)=sum(accurate(num,:))/3;
end


% %�޳���������
% cc=0;
% for i=1:3
%     for j=1:3000
%         if(abs(percent(i,j))>0.7)
%             percent(i,j)=0;
%             cc=cc+1;
%         end
%     end
% end







