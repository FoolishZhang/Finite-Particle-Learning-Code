# 固定K弱载荷下的边际不变依赖偏好：作者推导

状态：本轮作者推导与待核数值预测，不是外部模型已经审核的定理；未整合稿件。它不使用原n增长定理。下文K为固定正整数，m有限，偏置为0。所有角度只在固定紧邻域内变化。

## 对象和结论

令v_j(phi)=s_j(cos phi_j,sin phi_j)，s_j>0固定；W_j=\varepsilon v_j，Z~N(0,I_2)，观测自旋y_j∈{-1,1}。给定Z的条件质量为prod_j sigma(\varepsilon y_j v_j Z)。教师采用同一\varepsilon、相同行范数和角度phi_*。令c_ij(phi)=v_i^T v_j（i<j），c_*为教师向量。定义

\[
F_{K,\varepsilon}(\phi)=E_{Y\sim P_{\varepsilon,\phi_*}}E_{Z_1,\ldots,Z_K}
\left[-\log\left\{K^{-1}\sum_{k=1}^K\prod_j\sigma(\varepsilon Y_jv_j(\phi)^TZ_k)\right\}\right].
\]

则在保持s的角度类上，固定K时

\[
F_{K,\varepsilon}(\phi)-F_{K,\varepsilon}(\phi_*)
=\varepsilon^4\left[a_K(\|c(\phi)\|^2-\|c_*\|^2)-b_Kc_*^T(c(\phi)-c_*)\right]+O(\varepsilon^6),
\]

\[
a_K=\frac{(K-1)(K-2)}{32K^2},\qquad b_K=\frac{K-1}{16K}.
\tag{JD1}
\]

余项在角度紧邻域内连同所用的一、二阶角度导数一致。总体观察风险为

\[
R_{\varepsilon}(\phi)=D_{KL}(P_{\varepsilon,\phi_*}\Vert P_{\varepsilon,\phi})
=\frac{\varepsilon^4}{32}\|c(\phi)-c_*\|^2+O(\varepsilon^6).
\tag{JD2}
\]

沿任意光滑phi(t)、phi(0)=phi_*，记dot c=dc/dt(0)，得到

\[
\frac{d}{dt}F_{K,\varepsilon}(\phi(t))\bigg|_0
=-\frac{K-1}{8K^2}\varepsilon^4\sum_{i<j}c_{*,ij}\dot c_{ij}+O(\varepsilon^6),
\]

\[
R_{\varepsilon}'(0)=0,\qquad R_{\varepsilon}''(0)
=\frac{\varepsilon^4}{16}\sum_{i<j}\dot c_{ij}^2+O(\varepsilon^6).
\tag{JD3}
\]

K=1的F角度不变是精确身份；K≥2且sum c_* dot c>0时，足够小\varepsilon>0下，该方向严格降低有限目标而增加真值附近的观察风险。这是固定K、弱载荷结论，没有给可用于\varepsilon=.8的显式余项门。

## 系数推导

对固定y写标准化单粒子质量2^m p(y|Z)=1+\varepsilon h_1+\varepsilon² h_2+\varepsilon³ h_3+\varepsilon⁴ h_4+…，令x_j=v_j^TZ。有

\[
h_1=\tfrac12\sum_j y_jx_j,\quad h_2=\tfrac14\sum_{i<j}y_iy_jx_ix_j,
\]

\[
h_3=\tfrac18\sum_{i<j<l}y_iy_jy_lx_ix_jx_l-\tfrac1{24}\sum_jy_jx_j^3.
\]

h_4仅包含两个或四个不同标签的乘积，因此均匀标签下E_y h_4=0。以bar h表示K个粒子的平均，负log展开的四阶项是

\[
-\bar h_4+\bar h_1\bar h_3+\tfrac12\bar h_2^2-\bar h_1^2\bar h_2+\tfrac14\bar h_1^4.
\tag{JD4}
\]

均匀标签的Walsh正交性和Gaussian四阶矩给出下表；这里只列对每个i<j的c_ij²系数，其余项仅依赖固定行范数。

|JD4中的项|c_ij²系数|
|---|---:|
|−E bar h_4|0|
|E bar h_1 bar h_3|0（其总量为−sum_j s_j⁴/(16K)）|
|E bar h_2²/2|(K+1)/(32K)|
|−E bar h_1² bar h_2|−(K+1)/(8K²)|
|E bar h_1⁴/4|3/(16K²)|
|合计|(K−1)(K−2)/(32K²)|

关键可逐项核的矩为

\[
E(\overline{x_ix_j})^2=(1+K^{-1})c_{ij}^2+K^{-1}s_i^2s_j^2,
\]

\[
E(\bar x_i\bar x_j\overline{x_ix_j})
=K^{-2}\{s_i^2s_j^2+(K+1)c_{ij}^2\}.
\]

第二式仅有三个索引全同、或bar x_i和bar x_j共用索引而第三个独立的两类非零项。bar x的协方差为C/K，故

\[
E_yE_Z(\bar h_1^4)/4
=\frac3{64K^2}\{(\sum_j s_j^2)^2+4\sum_{i<j}c_{ij}^2\}.
\]

固定y的二阶负log项经银行平均为

\[
\frac1{8K}\sum_j s_j^2-\frac{K-1}{4K}\sum_{i<j}y_iy_jc_{ij}.
\tag{JD5}
\]

教师有E(Y_iY_j)=\varepsilon² c_*,ij/4+O(\varepsilon⁴)。因此JD5在教师下的角度相关四阶项为−(K−1)c_*^T c/(16K)。四阶均匀标签项只需在\varepsilon=0的教师下评价，得到JD1。教师概率的四阶修正乘以零阶常数m log2时求和为0；无额外角度项。

精确观察质量的展开为

\[
P_{\varepsilon,\phi}(y)=2^{-m}\{1+\tfrac{\varepsilon^2}4\sum_{i<j}y_iy_jc_{ij}(\phi)+O(\varepsilon^4)\}.
\]

在有限字母表上对KL展开，再用Walsh正交性，得到JD2。JD1微分后利用2a_K−b_K=−(K−1)/(8K²)即得JD3。

## 为什么余项与微分交换合法

不能对Gaussian无界输入直接宣称逐点泰勒余项一致。这里对完整负logmeanexp的实变量导数作控制：每个log sigma的高阶实导数有界，\varepsilon与角度导数只再引入固定有限次数的Gaussian坐标；logsumexp导数是有限阶加权累积量，归一权重位于[0,1]。固定m,K和角度紧集上，所需至\varepsilon六阶、附带至二阶角度导数均由常数乘有限次max_k||Z_k||的多项式支配，Gaussian各阶矩有限。可在期望下作有限阶Taylor并控制导数余项。

同时把所有Z_k换为−Z_k表明模型概率、教师和银行平均目标均为\varepsilon的偶函数。六阶Taylor控制因此给O(\varepsilon⁶)，并对角度微分保留该阶。精确概率在\varepsilon=0为2^-m，在一个小的实邻域中统一为正，故有限字母表KL可合法展开。这里不声称复解析全局性、K增长一致性或显式六阶常数。

## 一个没有独立可删块的预定方向

本包s_j=1，phi_*=(-.7,-.5,-.3,-.1,.1,.3,.5,.7)，phi(t)=(1−t)phi_*。sum phi_j=0，因此扰动与共同旋转正交。对每对i<j，Delta_ij=phi_*,i−phi_*,j，

\[
c_{*,ij}=\cos Delta_{ij}>0,\qquad
\dot c_{ij}=Delta_{ij}\sin Delta_{ij}>0.
\]

所以JD3的有限目标导数首项严格负。所有行非零，任意两行内积严格正，Gaussian依赖图完整，不存在可删的指定独立观测块。事实上Gaussian分部积分/Price导数表明，自旋成对矩对c_ij的导数为\varepsilon² E[sech²(\varepsilon X_i/2)sech²(\varepsilon X_j/2)]/4>0。因此每个\varepsilon>0下，沿任意足够小的正t，至少一个观察成对矩严格改变、联合分布不同，KL严格正；风险不是旋转不识别伪象。

角度压缩不保持整个联合观察律，只保持每个logit N(0,\varepsilon²)及每个观察Bernoulli(1/2)边际。完整学习仍允许全部b,W变化；该角度路径只提供一个可识别的依赖方向，并不强迫训练仅沿此方向移动。

## 限制与预先给出的可证伪读数

- 对固定K≥2的小\varepsilon，符号是作者解析结论；未给\varepsilon=.8或任意训练终态的解析阈值。
- K2四维Gaussian求积检验JD3导数/\varepsilon⁴是否趋向−sum c dot c/32，以及t=.05的有限目标下降与风险上升。跨阶吻合仅为数值资格。
- 理论预言固定边际的相关强化倾向；自由载荷训练可能同时收缩行长，二者不能混为一个机制。终点交叉评价负责区分目标排序与求解误差，不能仅凭训练风险差宣称机制已经解释全部路径。
- 这是一个固定低维潜变量实例，不推广原正谱底类、n增长尺度、任意优化器或后验适配提议。是否值得纳入论文须另列提案并由用户安排审查。


## K=2的第二条系数核对

可不经JD4的三索引计数，使用精确身份

\[
-\log\{(e^{A_1}+e^{A_2})/2\}
=-(A_1+A_2)/2-\log\cosh\{(A_1-A_2)/2\}.
\]

第一项在角度类上恒定。写A_1−A_2=\varepsilon L+\varepsilon² Q+O(\varepsilon⁴)，其中L=\frac12\sum_j y_j(x_{1j}−x_{2j})、Q=−\frac18\sum_j(x_{1j}²−x_{2j}²)。对未缩放Gram矩阵C=(v_i^T v_j)，有

\[
E Q^2=\operatorname{tr}(C^2)/16,\qquad
E_yE L^4=\tfrac34\{(\operatorname{tr}C)^2+4\sum_{i<j}c_{ij}^2\}.
\]

log cosh的四阶系数是EQ²/8−EyEL⁴/192；前者对sum c²的系数为1/64，后者同为1/64，恰好抵消。因此K2的a_2=0。教师相关只进入二阶项的标签期望：E_Y E L²=tr C/2+sum_ij E(Y_iY_j)c_ij，得到−\varepsilon⁴ c_*^T c/32，即b_2=1/32。这与一般K的公式独立相符，仍须保留上面的可积导数余项条件。
