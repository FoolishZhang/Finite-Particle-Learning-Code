# 共线点的完整二阶分解

2026-10-07，作者推导。它给出本轮程序检查的数学对象；不把浮点驻点当严格存在性证明。设计与固定门见DESIGN (historical reference not distributed)，内核为[rankone_stationary_kernel.py](../../../script/rankone_stationary_kernel.py)。

## 1. 模型与固定目标

有m个二元观察，z_k~N(0,1)独立，规范共线参数为W_j=(a_j,0)。给定标签y，令

\[
A_k=\prod_j \sigma(b_j+a_jz_k)^{y_j}[1-\sigma(b_j+a_jz_k)]^{1-y_j},
\quad \pi_k=A_k/\sum_l A_l,\quad
F_K(b,a)=-\mathbb E_{y\sim q,z_{1:K}}\log(K^{-1}\sum_k A_k).
\]

q是外部固定目标，不依赖待学参数。总体诊断时q为保存的教师PMF，经验问题时为原流的固定频率。a为载荷行范数；不能用原W第一列代替它。

记p_k=σ(b+a z_k)、r_k=y−p_k、D_k=diag[p_k(1−p_k)]。θ=(b,a)的条件对数似然得分和负Hessian为

\[
G_k=\binom{r_k}{z_kr_k},\qquad
B_k=\begin{pmatrix}D_k&z_kD_k\\z_kD_k&z_k^2D_k\end{pmatrix}.
\]

有限K的精确导数为

\[
g_{ba}=-\mathbb E\sum_k\pi_kG_k,
\qquad
H_{ba}=\mathbb E\sum_k\pi_k B_k
-\mathbb E\sum_k\pi_k(1-\pi_k)G_kG_k^\top
+\mathbb E\sum_{k\ne l}\pi_k\pi_lG_kG_l^\top. \tag{1}
\]

证明：对log-sum-exp求导，负Hessian等于加权条件负Hessian减去粒子得分的加权协方差。展开协方差得到(1)。最后一项保留不同粒子的交叉；不能在对y取期望后才做外积。

## 2. 无需K维Gaussian银行的Laplace积分

对每个y取任意正尺度c_y，x(z)=A_y(z)/c_y，u>0，并定义

\[
\Psi(u)=\mathbb E_z e^{-ux(z)},\quad
B_0(u)=\mathbb E_z x(z)e^{-ux(z)},\quad
M_G(u)=\mathbb E_z x(z)e^{-ux(z)}G(z).
\]

由1/S=∫e^{−uS}du及1/S²=∫u e^{−uS}du，对Gaussian离散节点(z_l,ω_l)，(1)中前三个构件可写为

\[
d_l=\omega_lx_l K\int_0^\infty e^{-ux_l}\Psi(u)^{K-1}du,
\quad
c_l=\omega_lx_l K(K-1)\int_0^\infty u e^{-ux_l}B_0(u)\Psi(u)^{K-2}du,
\]

\[
g_{ba,y}=-\sum_l d_lG_l,\qquad
H_{ba,y}=\sum_l d_lB_l-\sum_l c_lG_lG_l^\top+
K(K-1)\int_0^\infty u M_G(u)M_G(u)^\top\Psi(u)^{K-2}du. \tag{2}
\]

最后按q对label取平均。程序用v=log u积分，所以d积分多u因子，c与交叉积分多u²因子。单粒子K=1时g=−E G、H=E B；K=∞时

\[
H_{\infty,y}=\mathbb E_{\pi_y}B-\mathbb E_{\pi_y}GG^\top+
(\mathbb E_{\pi_y}G)(\mathbb E_{\pi_y}G)^\top ,
\]

其中π_y为给定y的归一后验。此处外积同样先于q平均。

若离散Gaussian权重的总质量M不是精确1，保持原权重的finite-K银行测度总质量为M^K。与上式导数一致的离散损失为

\[
F_{K,y}^{(\nu)}=-M^K\log c_y+
\int_{-\infty}^{\infty}
\{\Psi(e^v)^K-M^Ke^{-Ke^v}\}\,dv. \tag{3}
\]

这由逐银行Frullani恒等式积分得到。尺度c_y可依参数选择；精确恒等式整体与它无关。数值截断造成的尺度依赖属于积分误差。d质量应接近M^K，c质量不超过M^K(1−1/K)；K∞后验d质量为1。本轮不重归一Gaussian或目标权重，(3)使有限离散测度的损失、梯度与二阶导数一致；真实Gaussian为M=1。制造测试另用明显非单位M来核这个区别。

## 3. 从16维共线求解到24维完整模型

完整载荷写成W_j=(u_j,v_j)。在(u,v)=(a,0)，联合Gaussian先验的第二坐标反射使目标满足F(b,u,v)=F(b,u,−v)。因此g_v=0，H_(b,u),v=0，完整Hessian为

\[
H_{\rm full}=\operatorname{diag}(H_{ba},H_\perp),\quad
H_\perp=\mathbb E\{\sum_k\pi_kD_k-\sum_k\pi_k(1-\pi_k)r_kr_k^\top\}. \tag{4}
\]

不同粒子第二坐标独立零均值，使横向块中的跨粒子项消失。它们不在(1)的径向/bias块中消失，这也是两块不能共用同一公式的原因。

固定行范数的角度参数化W_j=a_j(cosφ_j,sinφ_j)给出

\[
H_\phi=\operatorname{diag}(a)H_\perp\operatorname{diag}(a)
-\operatorname{diag}(a\odot g_a),\qquad H_\perp a=g_a,\quad H_\phi\mathbf1=0. \tag{5}
\]

Ward关系可由Gaussian分部积分得到，也可由完整模型的共同旋转不变性对梯度求导得到。Hφ是有符号Laplacian，边权a_i a_j EΣπ_k(1−π_k)r_ki r_kj。其与(5)直接计算的差异是本轮数值诊断。

在真正驻点g_b=g_a=0，旋转切向量为(0,0,a)，对应Hperp零模。取V为a正交补的任意正交基，若

\[
\lambda_{\min}(H_{ba})>0,\qquad \lambda_{\min}(V^\top H_\perp V)>0, \tag{6}
\]

则完整模型除共同旋转轨道外的23方向都正。可取局部切片a^T v=0；旋转轨道与切片横截，(6)是切片上的严格二阶充分条件，给出模共同旋转的局部极小。反射只带离散等价，不添连续零模。本轮用确定性Householder基计算V，未把数值旋转方向加入正曲率数。

若只有g小而非精确0，不能直接援引该定理声称已证明精确驻点存在。应称“经预定数值资格的驻定/局部极小二阶签名”。更强存在性需要另作误差界、区间Newton或等效认证，本轮未完成。Hessian的求解阻尼也不属于(6)证据；只检查原始H。

## 4. 总体风险探针的含义

P仅拟合总体F8。观察风险R=KL(q_teacher||p_θ)与总体积分负对数似然只差固定教师熵，因此它们的(b,a)梯度相同。预定方向−g_R/||g_R||若非零，就是局部一阶风险下降方向。若P同时呈现(6)与F8驻定签名，则足够小的该方向步长理论上升高F8而降低R；本轮只以预定五个有限步长检查数值现象，保留C/D差异和所有不符合情况。

P使用已知教师，是oracle机制诊断；不是第五个独立数据重复，也不提供一个未知教师下的可部署更新。K64和积分目标在F8终点上未被求至驻定，其负曲率不能称为各自的驻点鞍点结论。所有局部结论都不提供全局最优、随机Adam吸引概率、训练因果分解或新的显著性保证。
