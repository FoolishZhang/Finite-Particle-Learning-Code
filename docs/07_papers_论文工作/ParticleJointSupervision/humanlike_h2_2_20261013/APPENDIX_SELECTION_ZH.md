# 附录 H. 早期有限选择及其负对照

## H.1. 选择条件与区域证书的职责

对$\mathcal M$的邻域$U$，定义$\Delta_U=\inf_{C\notin U}V(C)-\min_CV(C)>0$；令$e=\sup_C|F(C)-F(I)-wV(C)|$、$\zeta=\operatorname{osc}_C(\widehat F-F)$、$\eta=\widehat F(\widehat C)-\inf_C\widehat F(C)$。与任一$V$极小点比较，得到常见充分条件

$$
2e+\zeta+\eta<w\Delta_U\ \Longrightarrow\ \widehat C\in U.
\tag{H.1}
$$

因为$U$外输出相对该极小点的经验差至少为$w\Delta_U-2e-\zeta$，与$\eta$上界矛盾。仅写条件并不产生有限数值证书。本研究直接给整片低风险区域上的经验准则下界，再单独审计有限总体比较；不从随机重启或网格最小值估计全局$e$、$\zeta$或优化regret。

## H.2. 连续低风险集合的同时下界

本节固定$m=3$、$\mu=0$、$C_*=I$、$\delta=.1$及$K=2$。令$q_y$为任意经验活跃标签频率，$Z_{bk}\in\mathbb R^3$及正无关权重$a_{bk}$为任意固定有限粒子库输入。对称根$S(C)=C^{1/2}$耦合候选。只去掉与候选无关的加法常数后，

$$
\widehat F(C)=\frac1B\sum_{b=1}^{B}\sum_yq_y\phi_y(S(C)Z_b),\qquad
\phi_y(X)=-\log\sum_{k=1}^K a_{bk}\prod_{i=1}^m\sigma(y_iX_{ki}),\qquad
a_{bk}=\frac{e^{S_{bk}^{\rm nuis}}}{\sum_\ell e^{S_{b\ell}^{\rm nuis}}}.
\tag{H.2}
$$

无关权重仍是随机似然因子，没有像FACT那样解析移除。取$r_0=10^{-6}$，令$\mathcal G_{r_0}=\{C\in\mathcal E_{.1}^{(3)}:\mathcal R(C)\le r_0\}$。Price、$\log\cosh u\le u^2/2$和Jensen给$\kappa'(c)\ge\beta=e^{-1/2}/4$。对$y_iy_j$使用Pinsker，并用真实pair矩为零，得到$|\kappa(c_{ij})|\le\sqrt{2r_0}$。因此

$$
C\in\mathcal G_{r_0}\ \Longrightarrow\
\|C-I\|_F\le d:=\frac{\sqrt{12r_0}}{\beta}=.02284535206697714\ldots,
\qquad\lambda:=1-d>0.
\tag{H.3}
$$

对$D=C-I$，$I+tD$上各矩阵最小特征值至少为$\lambda$。根身份求导给$SS'+S'S=D$和$SS''+S''S=-2(S')^2$；Sylvester逆算子界给

$$
\|S'\|_F\le\frac{\|D\|_F}{2\sqrt\lambda},\qquad
\|S''\|_F\le\frac{\|D\|_F^2}{4\lambda^{3/2}},\qquad
\|\nabla_X\phi_y\|_2\le\sqrt m,\quad
\|\nabla_X^2\phi_y\|_{\rm op}\le m+\tfrac14.
\tag{H.4}
$$

最后一界由加权发射Hessian（范数至多$1/4$）与log-sum-exp梯度协方差（范数至多$m$）得到。若$a_b^2=\sum_k\|Z_{bk}\|^2$、$A_2=B^{-1}\sum_ba_b^2$，链式法则再用$B^{-1}\sum_b a_b\le\sqrt{A_2}$控制二阶方向导数。记$g$为$I$处三个自由坐标导数，Taylor公式给

$$
H(A_2)=\frac{(m+1/4)A_2}{4\lambda}
+\frac{\sqrt{mA_2}}{4\lambda^{3/2}},\qquad
L_r=\frac{\|g\|_2d}{\sqrt2}+\frac{H(A_2)d^2}{2},\qquad
\inf_{C\in\mathcal G_{r_0}}\widehat F(C)\ge\widehat F(I)-L_r.
\tag{H.5}
$$

因子$1/\sqrt2$将自由非对角坐标转换为Frobenius范数。该界同时覆盖连续低风险集合的每个矩阵，无论是否落在搜索网格。于是，

$$
\widehat g:=\widehat F(I)-\widehat F(C_b)-L_r>0,\qquad
\widehat F(\widehat C)\le\widehat F(C_b)
\quad\Longrightarrow\quad \mathcal R(\widehat C)>r_0.
\tag{H.6}
$$

## H.3. 单独的有限总体比较

真实均匀标签分布及Gaussian先验使$F(C)$对$C$的同时符号共轭不变，故两无关条件下$I$的自由坐标梯度都为零。对(H.4)取期望，用$Ea_b^2=Km=6$和$Ea_b\le\sqrt6$，得到$A_2=6$时的界：

$$
\inf_{C\in\mathcal G_{r_0}}F(C)-F(C_b)
\ge g-\frac{H(6)d^2}{2},\qquad
g=F(I)-F(C_b),\qquad \frac{H(6)d^2}{2}=.00158844598226789635\ldots.
\tag{H.7}
$$

独立审计只估这一预定点差；用真实$1/8$权重精确平均八个活跃模式，这些权重不进入学习器。若$D_b=h_b(I)-h_b(C_b)$、$A=\|C_b^{1/2}-I\|_{\rm op}=\max(1-\sqrt{.1},\sqrt{2.8}-1)$，Lipschitz性给$|D_b|\le\sqrt3 A\max_k\|Z_{bk}\|$。取$T=8$、$E_b=\{\max_{k,i}|Z_{bki}|\le T\}$、$D_b^t=D_b1_{E_b}$及$M=3AT$，Gaussian Mills和Cauchy界给

$$
|D_b^t|\le M=16.410533615595888\ldots,\qquad
|ED_b-ED_b^t|\le t_T:=\sqrt3 A\sqrt{6\cdot12\varphi(T)/T}
<2.526\times10^{-7}.
\tag{H.8}
$$

对$L=2097152$个独立银行，缩放后的双侧经验Bernstein不等式[@maurer2009, Theorem 4]给下列半宽。$s^2$为$D_b^t$的无偏样本方差，$\alpha_j=.005$；在iid抽样模型及所述数值余量下，对两条件联合的总错误概率至多$.01$：

$$
b_L=\sqrt{\frac{2s^2\log(4/\alpha_j)}{L}}
+\frac{7(2M)\log(4/\alpha_j)}{3(L-1)}+t_T+10^{-9},
\qquad g\in[\overline D^t-b_L,\overline D^t+b_L].
\tag{H.9}
$$

## H.4. 固定程序、输入与完整结果

单份训练数据含独立教师生成的65536个iid三位观测。无关标签用Gaussian符号对称性积分，而每个银行仍保留$n$个随机无关似然因子。每个相关坐标取$\{-9,-6,-3,0,3,6,9\}/10$；以整数检查$10(C-.1I)$全部主子式，剩143个可行候选。double最小分数$10^{-12}$以内的词典序首点作提案；区间比较确认其不差于固定参照$(.9,.9,.9)$，否则返回参照。输出锁定后才算风险和谱；不以真实相关方向或风险oracle选择候选。

PCG64DXSM SeedSequence为`[202610060101,role]`：role0/1/2依次为训练标签、活跃Gaussian银行和无关Gaussian因子，3/4为独立审计的活跃/无关Gaussian。按[银行,粒子,坐标]顺序，每块4096生成。两条件共享标签和活跃prior；$n=0$无关权重设为$1/2$，$n=256$在同活跃输入上保留256个独立无关因子。训练银行65536，独立审计银行2097152。执行后没有更换种子、维数、粒子数、风险阈值或统计家族。共享输入只计一次：4325376个prior向量，各含3个活跃和256个无关Gaussian坐标，以及65536个活跃观测模式；无神经训练或梯度优化更新。

实现的界如下，单位为nats/观测：

|n|Empirical point gap|Linear cost|Quadratic cost|Empirical region lower|
|---|---|---|---|---|
|0|.003594355617|.000015488961|.001585398366|.001993468290|
|256|.004741770692|.000019724039|.001585398366|.003136648287|

独立点差审计给出：

|n|Population point mean|99% family interval|Population region lower|
|---|---|---|---|
|0|.002478573642|[.001864904388, .003092242896]|.000276458406|
|256|.003266233361|[.002280561286, .004251905435]|.000692115304|

显示值作了舍入，所有正性判定使用未舍入区间。两提案本来就等于$C_b$，无回退或比较后的改选。两输出谱为$(.1,.1,2.8)$，数值观察KL为$.031103092779$，真实独立参照风险为零。解析证书只证明$\mathcal R>10^{-6}$，不证明$.0311$的紧下界。尤其$n=0$控制也排除了低风险区域。本研究为指定程序分辨有限错误选择，但没有识别新增无关因素为必要原因，没有确定完整全局最小集，也未建立常见随机优化器失效。

## H.5. 数值核验与保留的故障

制造输入将double评分和自由坐标导数与128-bit Arb球算术比较，评分/梯度差小于$3\times10^{-17}$，中心有限差分梯度误差小于$2.4\times10^{-11}$。保存训练库的每个银行均在$I$、$C_b$和返回点作球算术重算，并重算$I$梯度。对该实现程序证书，binary64银行输入作为精确数处理。对正定的舍入对称根$S_0$，球残差界给$\|S_0-C^{1/2}\|_F\le\|S_0^2-C\|_F/\sqrt\delta$；分数余量至多为该界乘$\sqrt{mA_2}$。$S_0$正定性由Sylvester主子式核对。

总体审计保存全部配对银行差并重算均值/方差，每条件预定32个等距银行以Arb核对。它们为(H.9)的$10^{-9}$显式数值余量提供资格检查，但不证明全局浮点误差界，也不是每个总体节点的区间求积；因此总体覆盖声明保留这一数值条件。确定性经验区域证书不依赖这项抽查。

选择后，观测概率使用精确的中心化三自旋表达$Q_C(y)=\{1+\sum_{i<j}\kappa(c_{ij})y_iy_j\}/8$，奇数阶自旋矩由对称性为零。pair Gaussian求积32/48/64阶另对照独立三维32/48阶；最大概率差小于$3.21\times10^{-13}$，风险变化小于$3.23\times10^{-13}$，质量误差小于$1.12\times10^{-16}$。这是数值资格，不是严格Gaussian积分余项证书；区域证明依赖Price与Pinsker，而不是求积误差声明。

原运行完成数据、网格提案、总体逐银行差和全量区间选择证书后，在方差汇总的NumPy布尔值写盘处失败。另行冻结的接续修正标量类型，仅以同一银行和已锁定选择完成未保存的诊断。原错误、保存输入/输出及恢复成本均保留，未新增数据、prior或学习输出。计算使用NumPy/SciPy double，球算术使用python-flint0.8.0的128-bit Arb。原无关Gaussian坐标可由固定RNG重建，其充分log和、活跃银行、标签和逐银行差保存在本地；哈希不能替代这些数组。
