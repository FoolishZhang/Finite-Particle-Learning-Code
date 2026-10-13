# 同总对数方差下的粒子几何：半数保留与单粒子支配

日期：2026-10-02。本页解释已预定记录的noise-only ESS诊断，零新增RNG、积分、拟合或统计门。结论仅针对无关似然因子的归一化粒子权重；包含active似然后的完整权重是另一个对象。

给定一条固定无关观测，可由Gaussian符号对称性取其噪声权重分布的全正代表。令K个独立权重为\(W_{n1},\ldots,W_{nK}>0\)，并定义

$$
\pi_{nk}=\frac{W_{nk}}{\sum_lW_{nl}},\qquad
\operatorname{ESS}_{n,K}=\frac1{\sum_k\pi_{nk}^2}
=\frac{(\sum_kW_{nk})^2}{\sum_kW_{nk}^2}.
\tag{W.1}
$$

同一粒子数组用于不同观测或条件时，它们的ESS读数通常相关；这里没有把这些复用当独立样本。

## 1. 集中族的权重极限与四阶矩

将[集中族](CONCENTRATED_THEORY.md)的强因子Gaussian重标为G，写为

$$
W_n=U_n\,2\sigma(A_nG),\qquad
U_n=\prod_{i=1}^{n-1}2\sigma(G_i/n),
\tag{W.2}
$$

其中G独立于弱因子。为陈述跨n的\(L^2\)收敛，可选取一个保持G相同的耦合；每个n的权重边缘分布仍与原构造相同。这不要求实验在不同维数使用同一个强因子坐标。

由\(EU_n=1\)及集中理论的二阶矩界，

$$
E(U_n-1)^2=EU_n^2-1
\le\exp\!\left(\frac{n-1}{4n^2}\right)-1\to0.
\tag{W.3}
$$

又\(A_n\to\infty\)，故\(2\sigma(A_nG)\to2\,1_{\{G>0\}}\)几乎处处，并由有界性在\(L^2\)中收敛。因此

$$
\|W_n-2\,1_{\{G>0\}}\|_2
\le2\|U_n-1\|_2+
\|2\sigma(A_nG)-2\,1_{\{G>0\}}\|_2\to0,
\quad EW_n^2\to2.
\tag{W.4}
$$

后一矩极限只依赖边缘分布，不依赖所选跨n耦合。

为了控制样本平方均值，还需四阶矩。令\(T=\tanh(G/(2n))\)，其奇数矩为零，且

$$
E(1+T)^4=1+6ET^2+ET^4
\le1+\frac{3}{2n^2}+\frac{3}{16n^4}.
\tag{W.5}
$$

使用强因子至多为2及独立性，得到

$$
EW_n^4\le16\exp\!\left(\frac{3}{2n}+\frac{3}{16n^3}\right)
\le16e^{27/16}.
\tag{W.6}
$$

## 2. 集中族在增长K下保留线性数量的有效粒子

令n→∞且\(K_n\to\infty\)，不要求K_n为特定增长阶。对每个n，粒子之间独立；跨n怎样耦合不影响下面的依概率结论。记\(\overline W_n=K_n^{-1}\sum_kW_{nk}\)、\(\overline{W_n^2}=K_n^{-1}\sum_kW_{nk}^2\)。二阶及四阶矩界分别给

$$
\overline W_n\to_p1,\qquad
\overline{W_n^2}\to_p2,\qquad
\boxed{\operatorname{ESS}_{n,K_n}/K_n\to_p1/2.}
\tag{W.7}
$$

前两项由Chebyshev及\(EW_n^2\to2\)得到，最后使用(W.1)和连续映射。这里的“一半”是渐近noise-only ESS比例，并非固定K下确定地剩余K/2个粒子。

令\(M_2=\sup_nEW_n^2<\infty\)。对任意ε>0，分开\(\overline W_n<1/2\)事件并对单个大权重用union bound，有

$$
\begin{aligned}
P(\max_k\pi_{nk}>\varepsilon)
&\le\frac{4(M_2-1)}{K_n}
+K_nP(W_n>\varepsilon K_n/2)\\
&\le\frac{4(M_2-1)}{K_n}
+\frac{4M_2}{\varepsilon^2K_n}\to0.
\end{aligned}
\tag{W.8}
$$

所以集中族没有单个粒子的渐近支配。若K固定，则强因子全为正的事件仍有概率\(2^{-K}>0\)，在该事件上权重均趋2且ESS趋K；因此不能把(W.7)扩张成固定K下ESS趋于确定的K/2。

## 3. diffuse与有界异质族的相反极限

对尺度固定为1的diffuse族，或[有界异质族](HETEROGENEOUS_THEORY.md)中任何确定数组\(\boldsymbol a_n\in[a_-,a_+]^n\)，记
\(s_n^2=\sum_i v(a_{ni})\)。
考虑\(2\le K_n\le C_0n^p\)。noise-only的softmax中active分数A全部为0，因此已证相对粒子对核可以直接使用，不需要再截断active分数：

$$
\begin{aligned}
E\!\left[1-\sum_k\pi_{nk}^2\right]
&=K_n(K_n-1)E[\pi_{n1}\pi_{n2}]\\
&=\frac{J_{K_n}}{s_n}\,[1+o(1)]\longrightarrow0.
\end{aligned}
\tag{W.9}
$$

该近似对固定正紧区间内全部数组及允许K一致；\(s_n\asymp\sqrt n\)、\(J_{K_n}=O(\sqrt{\log n})\)保证右端趋零。左侧非负，故Markov给\(\sum_k\pi_{nk}^2\to_p1\)。再用
\(\sum_k\pi_{nk}^2\le\max_k\pi_{nk}\le1\)，得到

$$
\boxed{\operatorname{ESS}_{n,K_n}\to_p1,\qquad
\max_k\pi_{nk}\to_p1.}
\tag{W.10}
$$

特别是K_n=n+2时，\(\operatorname{ESS}_{n,K_n}/K_n\to_p0\)。
这里没有从概率收敛推出增长K下\(E\,\operatorname{ESS}\to1\)：ESS的上界K也增长，现有论证没有给这一随机量的一致可积性。

## 4. 对有限实验的解释边界

两族可以具有相同的总\(\operatorname{Var}(\log W)=n v(1)\)，却呈现不同的noise-only权重几何：集中族在增长K下的有效粒子数占比趋1/2；diffuse及固定正紧区间内的异质族趋单粒子支配。差异不由总对数方差一个数决定。

这些是规定序列上的概率结论，不是当前有限n的精确ESS公式、期望ESS结论或新统计成功门。一个active似然因子还会再次改变权重，不能把noise-only的1/2直接赋给完整似然ESS。ESS也不能替代对数尾控制：真实raw风险恢复仍需要集中理论和学习桥中的可积性与目标误差论证。预定MC诊断只用于观察这种结构差异，没有新增或追认任何原门。
