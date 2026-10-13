# 三个观察矩的联合风险外包与有限总体近优证书

这是一项事后解析加强，假设与原NuisanceContrast-v1相同，不修改原实验的预定判定。所有对数为自然对数。

## 1. 精确分布与低风险外包

设X为中心三维Gaussian，协方差C满足diag(C)=1、C≥.1I；给定X，三个Y_i∈{−1,1}条件独立，P(Y_i=1|X)=σ(X_i)。记κ(c)=E[tanh(X_i/2)tanh(X_j/2)]，其中这两个标准Gaussian的相关为c。独立教师P=Q_I在八个标签上均匀，风险为R(C)=D(P∥Q_C)。

中心反演X→−X以及logistic对称性给Q_C(y)=Q_C(−y)，全部奇数阶spin矩为零。二元Walsh反演因此精确给出

$$
Q_C(y)=\frac{1+u_y}{8},\qquad
u_y=\kappa(c_{12})y_1y_2+\kappa(c_{13})y_1y_3+\kappa(c_{23})y_2y_3.
\tag{1}
$$

不是小相关近似。三个二阶Walsh角色关于P正交：

$$
E_Pu=0,\qquad E_Pu^2=\sum_{i<j}\kappa(c_{ij})^2,\qquad
R(C)=-E_P\log(1+u).
\tag{2}
$$

若R(C)≤r0，考察事件A_y={y,−y}。P(A_y)=1/4，Q_C(A_y)=(1+u_y)/4。使用原方向D(P∥Q_C)的Pinsker界，得到

$$
|u_y|/4\le\operatorname{TV}(P,Q_C)\le\sqrt{r_0/2},
\qquad |u_y|\le a:=\sqrt{8r_0}.
\tag{3}
$$

取a<1。对正负u均成立的精确积分余项是

$$
-\log(1+u)=-u+u^2\int_0^1\frac{1-t}{(1+tu)^2}\,dt
\ge-u+\frac{u^2}{2(1+a)^2}.
\tag{4}
$$

由(2)平均掉线性项，R(C)≥Σκ(c_ij)²/[2(1+a)²]。Price公式、log cosh v≤v²/2与Jensen给全局导数下界

$$
\begin{aligned}
\kappa'(c)&=\tfrac14 E\{\operatorname{sech}^2(X_i/2)\operatorname{sech}^2(X_j/2)\}\\
&\ge\tfrac14 E\exp\{-(X_i^2+X_j^2)/4\}
\ge e^{-1/2}/4=:\beta.
\end{aligned}
\tag{5}
$$

因κ(0)=0，|κ(c)|≥β|c|，包括负c。于是

$$
R(C)\le r_0\quad\Longrightarrow\quad
\|C-I\|_F^2=2\sum_{i<j}c_{ij}^2
\le\frac{4(1+\sqrt{8r_0})^2r_0}{\beta^2}=:d_{\rm new}^2.
\tag{6}
$$

原外包分别对三个相关使用Pinsker再求和，平方半径是12r0/β²；新界利用同一个联合分布的三个正交矩，平方半径比为(1+sqrt(8r0))²/3。这不是新增数据带来的置信区间缩窄。

## 2. 对整个连续区域的评分下界

令S(C)=sqrt(C)，每银行最大活跃粒子范数M=max_k||Z_k||。无关权重a_k正且与C无关。对任意标签和C(t)=I+tD，定义ℓ_k=Σ_i logσ(y_i(S(C(t))Z_k)_i)、φ=−logΣ_k a_k expℓ_k。归一后的完整权重为π，有

φ''=−Σπ_kℓ_k''−Var_π(ℓ_k')。

logistic的一阶、二阶导数界和Sylvester平方根导数界给

|φ''|≤[(13/4)M²/(4λ)+sqrt(3)M/(4λ^(3/2))]||D||_F²，

其中λ=1−d_new>0，因为路径最低特征值≥1−||D||_op≥1−d_new。原最大范数曲率证明完整保留在[前轮推导](../NuisanceContrast_v1_20261006/DERIVATION.md)；本轮只将其适用球半径缩小。

银行平均或总体期望用Jensen控制E M≤sqrt(E M²)。记

$$
H(U;\lambda)=\frac{(13/4)U}{4\lambda}+\frac{\sqrt{3U}}{4\lambda^{3/2}},\qquad
b_K=\tfrac83(\log K+3\log2).
\tag{7}
$$

Gaussian指数矩在t=3/8给E M²≤b_K。总体真标签均匀、Gaussian标准输入在逐坐标变号下不变，因此F_n(D_s C D_s)=F_n(C)，D_s为任意对角符号矩阵；每个自由相关方向的一阶导数在I为零。上述可积导数界允许在期望内求导。

对于两个权重条件分别成立

$$
\inf_{R(C)\le r_0}F_n(C)\ge F_n(I)-c_{\rm new},\qquad
c_{\rm new}=\tfrac12H(b_K;1-d_{\rm new})d_{\rm new}^2.
\tag{8}
$$

经验准则的标签频率固定，未必具有这个驻点对称性。保留原三自由坐标梯度g_n(I)，下界为Fhat_n(I)−||g_n(I)||d_new/sqrt2−H(U;1−d_new)d_new²/2。U由原全量Arb范数上界提供，不重新抽银行。

## 3. 同一个覆盖事件与事后推论

原审计已同时覆盖g0=F0(I)−F0(Cb)、g256=F256(I)−F256(Cb)、Δ=g256−g0。令E表示该99%联合覆盖事件，继承原iid抽样和数值资格条件，L256为原g256下端点。由(8)，在E上

$$
\inf_{R(C)\le r_0}\{F_{256}(C)-F_{256}(C_b)\}
\ge L_{256}-c_{\rm new}.
\tag{9}
$$

这是同一固定目标和锚点的确定性推论。数据促使寻找更紧的正确证明，不另花alpha：选用任何已证明有效的确定性界，结论仍逐点成立于同一个E。本轮没有择优挑选未同时覆盖的随机比较、没有从新数据估计常数、没有改变风险门。若改变这些对象，需要重新处理选择与覆盖。

原预定门用旧代价，仍然失败。新结论必须标记为事后解析加强，不能将原结果改成七门全过。原总体仅抽查32银行的数值资格局限也完整保留；更紧的解析外包不证明所有总体评分节点的1e−9误差预算。

## 4. 一个有限总体近优误差门

若经向外区间计算得到η=L256−c_new>0，则整个R≤r0集合被可行点Cb严格支配。对参数类中的任何C，若

$$
F_{256}(C)-\inf_{C'\in\mathcal E_{.1}^{(3)}}F_{256}(C')<\eta,
\quad\text{则}\quad R(C)>r_0.
\tag{10}
$$

证明：若C低风险，(9)使F256(C)≥F256(Cb)+η，而总体下确界不大于F256(Cb)，与近优误差小于η矛盾。目标在紧参数类上连续且有限：逐银行损失位于[0,3log2+3M]内，这个包络可积；所以全局最小点存在，也全部在低风险集合外。

这不声称Cb就是全局最小点，不确定完整极小集，也不保证任意优化器或原有限搜索满足总体regret门。n0的信息仍只有实际程序返回I、原总体锚点偏好I等事实，未证明I是n0连续总体目标的全局最优。

限定条件是m=3、零均值、独立教师、固定K100及既有审计。它不是主定理高维渐近余项认证，不直接推广到非均匀教师、增长活跃维数或后验自适应提议。Walsh恒等式、Pinsker、Price、Taylor和Gaussian矩界均为经典工具；贡献在该实例中形成更有力、可计算的有限总体后果。
