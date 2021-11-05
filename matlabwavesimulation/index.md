# Matlab波浪数值模拟


**写在前面**：最近几天新冠病毒疫情还未平息，在家帮女票研究波浪模型的时候探索了一下用Matlab进行波浪数值模拟的简单方法，在这里写一个简单教程，因为我在网上也没有找到写的比较完整的波浪模拟代码，所以来这里占个坑，希望对大家有所帮助。

## 理论依据

详情参考这两篇论文：

> [1] 刘素美. 波浪数值模拟[J]. 科技与创新, 2018(13):132-133.
> 
> [2] 赵珂,李茂华,郑建丽,田冠楠. 基于波浪谱的三维随机波浪数值模拟及仿真[J]. 舰船科学技术, 2014,36(02):37-39.

简单来说就是，波浪的模拟，可以由不同方向角、不同频率的很多个波，用随机的初始相位初始化后叠加得到。显然，组成波浪的波的频率个数越多、方向角个数越多，能够形成的波就更复杂（直观上也更真实）。

**先给出波面模型的公式**（根据文献[1]）
$$ z = \eta(x, y, t)=\sum_{i=1}^{M} \sum_{j=1}^{N} \zeta_{i j} \cos [{k}_{i}\left(x \cos \alpha_{d j}+y \sin \alpha_{d j}\right)-\omega_{di}t+\beta_{ij} $$
其中$\zeta_{ij},k_{i},\alpha_{dj},\omega_{di},\beta_{ij}$分别为波浪的波幅、波数、方向角、频率和相位角。且$k_{i} = \omega_{di}^{2}/g$。

可以看出，其中$x,y,t$是需要输入的参数，即坐标位置和时间，$z$就是波面高度，由于波的形成需要如上的这些参数。其中方向角和频率是需要进行划分的，其余的参数全都可以由不同的方向角和频率计算得到，下面讲波浪参数的确定。

### 方向角划分和选取

对传播方向角$\alpha$进行划分时，设方向角的变化范围为主波向$\alpha_{main}$两侧$[-\pi/2,\pi/2]$的范围，将此区域$N$等分，每一份的宽度为$d\alpha=\pi/N$（论文这里写的有误，请大家注意），选取每段的中心值作为方向角$\alpha_{dj}$。

### 频率的划分和选取

论文中给定了频率选取区间的计算方式，这里不做描述，直接关注频率的划分公式

$$\omega_{i}=\left[\frac{3.11}{H_{1 / 3}^{2} \ln (M+2 / i)}\right]^{1 / 4}$$

$$\omega_{di}=\frac{\omega_{i+1}+\omega_{i}}{2}$$

这里采用等分能量法分割频率，将频率划分为$M$个等能量的区间，$\omega_{i}$是各区间的分界频率，$\omega_{di}$是各频率区间的中心值作为选取的频率。

**备注：原文中第一个公式中是$M/i$，因为i的取值只能是从1~M-1，所以只能产生M-2个区间，这里坐了一下修改，保证M的个数与选取的频率个数相同**


### 利用三维随机波浪谱确定波幅$\zeta_{ij}$

标准波浪谱为PM谱

$$S_{PM}(\omega)=\frac{0.78}{\omega^{5}}exp[-\frac{3.11}{\omega^{4}H_{1/3}^{2}}]$$

其中$H_{1/3}=0.0214v^{2}$为有义波高，$v$是海面风速，会影响波浪高度。由于PM 谱描述的是能量随频率的变化，而对于三维随机波浪，其能量分布与频率和方向角都有关，并且认为频率和方向角的影响相互独立，则引入只与方向角$\alpha$有关的方向扩展谱函数$D_{f}(\alpha)$

$$D_{f}(\alpha)=\frac{2}{\pi} \cos ^{2} \alpha,\left(-\frac{\pi}{2} \leqslant \alpha \leqslant \frac{\pi}{2}\right)$$

最终得到三位随机波浪的方向波谱：

$$S_{3D}(\omega,\alpha)=S_{PM}(\omega)D_{f}(alpha)$$

将划分好的频率和方向角代入下式，即可得到每个单元组成波的波幅$\zeta_{ij}$

$$\zeta_{ij}=\sqrt{2S(\omega,\alpha)d{\omega}d{\alpha}}$$

**备注：论文这里公式有误，我已经做了修改**

### 随机初始相位$\beta_{ij}$的生成

文中写的是线性乘同余法，其实用简单的$0-2\pi$的均匀分布随机采样就可以。

## 基本实现的静态波浪生成代码

```matlab
n = 64;
map = zeros(n,n);
M = 1;    % ferequence number
N = 50;   
beta = 2*pi*rand(M,N);  

for x = 1:n
    for y=1:n
        map(x,y) = bo(x,y,M,N,beta);
    end
end

XX = 1:n;
YY = 1:n;
surf(XX, YY, map);
axis([-5, n+5, -5, n+5, -5, 5])

function H = bo(x,y,M,N,beta)
t = 0;     
v = 5;    
g = 9.8;  
H_value = 0.0214*v^2;  
alpha_main = 0;        

da = pi/N; 
a = alpha_main-pi/2+da/2 : da : alpha_main+pi/2-da/2; 

wi = (3.11./(H_value^2*log((M+2)./(1:M+1)))).^(1/4);
% wi = zeros(M+1,1);
% for i = 1:M+1
%     wi(i) = (3.11/(H_value^2*log((M+2)/i)))^(1/4);
% end 

w = (wi(2:end)+wi(1:end-1))/2;
dw = wi(2:end)-wi(1:end-1);

% w = zeros(M,1);
% dw = zeros(M,1);
% for i = 1:M
%     w(i) = (wi(i+1)+wi(i))/2;
%     dw(i) = wi(i+1)-wi(i);
% end

 H = 0;
for i=1:M
    for j=1:N
        adj = a(j);
        wdi = w(i);
        Spm_w = 0.78*exp(-3.11/(wdi^4*H_value^2))/wdi^5;
        Df_alpha = 2*(cos(adj)^2)/pi;
        S3d = Spm_w*Df_alpha;
        A = sqrt(2*S3d*dw(i)*da);
        ki = wdi^2/g;
        H = H + A*cos(ki*(x*cos(adj)+y*sin(adj))-wdi*t+beta(i,j));
    end
end
end
```
![静态波浪](/images/blogimgwave.svg "静态波浪")

## 波浪的动态实现
**（计算已改为矩阵并行化，为了提升实时的渲染速度）**
```matlab
n = 100;  
t = 1;  
v = 8;  
M = 15; 
N = 15; 

g = 9.8; 
H_value = 0.0214*v^2; 
alpha_main = 0;  

beta = 2*pi*rand(M,N);  

da = pi/N; 
a = alpha_main-pi/2+da/2 : da : alpha_main+pi/2-da/2; 

wi = (3.11./(H_value^2*log((M+2)./(1:M+1)))).^(1/4);

% wi = zeros(M+1,1);
% for i = 1:M+1
%     wi(i) = (3.11/(H_value^2*log((M+2)/i)))^(1/4);
% end 

w = (wi(2:end)+wi(1:end-1))/2;
dw = wi(2:end)-wi(1:end-1);

% w = zeros(M,1);
% dw = zeros(M,1);
% for i = 1:M
%     w(i) = (wi(i+1)+wi(i))/2;
%     dw(i) = wi(i+1)-wi(i);
% end

Spm_w = 0.78*exp(-3.11./(w.^4*H_value^2))./w.^5;
Df_alpha = 2*(cos(a).^2)/pi;
S3d = Spm_w'*Df_alpha;
A = sqrt(2*S3d.*(dw'*da));
ki = w.^2/g;

A = repmat(A(:)', n*n, 1);
aj = repmat(a, M, 1);
a = aj(:)'；
ki = ki';
ki = repmat(ki, 1, N);
k = ki(:)'; 
extend_w = repmat(w',1,N);
extend_w = repmat(extend_w(:)', n*n, 1);
beta = repmat(beta(:)', n*n, 1);


XX = 1:n; YY = 1:n;
[X,Y] = meshgrid(XX, YY);
t0 = 0;
dt = 0.1；
T = 100;

X = reshape(X, n*n, 1);
Y = reshape(Y, n*n, 1);

clf
shg
set(gcf);
MAP = wave2(X,Y,A,k,a,extend_w,t0,beta);
MAP = reshape(MAP, n, n);
surfplot = surf(XX, YY, MAP);
%shading interp
axis([-5,n+5,-5,n+5,-10,10])

for t = 0:dt:T
    MAP = wave2(X,Y,A,k,a,extend_w,t,beta);
    MAP = reshape(MAP, n, n);
    set(surfplot,'zdata',MAP);
%     shading interp
    drawnow
end


function wave = wave2(X,Y,A,k,a,extend_w,t,beta)
wave = sum(cos((X*cos(a)+Y*sin(a)).*k-extend_w.*t+beta).*A,2);
end
```
![动态波浪](/images/blogimgwave.gif "动态波浪")

