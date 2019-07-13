clc
clear
a=load('copy_error.mat');
y=zeros(1,100);
x=0.01:0.01:1;
% x2=0.01:0.01:1;
% x3=0.01:0.01:1;
for i=1:100
    y(i)=a.b.res_error(i,4);
    if(i>90)
        if(y(i)>0.7984)
             y(i)=0.7983;
        end
    end
    
end
plot(x,y,'r-');