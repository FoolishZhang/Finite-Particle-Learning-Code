# R03：破坏反序对称的教师族与连续根证书接口

日期：2026-10-11。本文是作者内部推导和实现接口，不是外部科研审核，也不是已经完成的 R03 参数区间证书。推导不运行真实教师/候选积分、不求新根、不读取科学 NPZ；旧稿 v1.9.1、原根证书与 R01 原文件保持。实际通过范围必须另由冻结规则、生产区间和逐段收据给出。

来源是原 [WITNESS_ARGUMENT](../CertifiedF8Hessian_v1_20261008/WITNESS_ARGUMENT.md)、GEOMETRY (historical reference not distributed)、三块 Hessian 恒等式 (historical reference not distributed)和 R01 角风险推导 (historical reference not distributed)。本文将其扩成八个自由共线载荷及参数区间的条件判据；不把旧四变量数值矩阵当作新八变量认证。

## 1. 对象与需要认证的结论

固定 m=K=8、潜变量维数2。教师偏置恒为零，行长 rho=4/5，角度

\[
\phi=(-7/10,-1/2,-3/10,-1/10,1/10,3/10,1/2,7/10),
\quad W_{*,j}(\lambda)=\rho\bigl(\cos(\phi_j+\lambda\mathbf1_{j=1}),
\sin(\phi_j+\lambda\mathbf1_{j=1})\bigr).
\tag{C1}
\]

令 q_lambda 为这个精确教师的观察 PMF。候选仍用完整 Cartesian 参数 theta=(b,u,v)，目标与风险为

\[
 F_\lambda(\theta)=-\sum_yq_\lambda(y)E\log\left(K^{-1}\sum_{k=1}^K A_y(Z_k;\theta)\right),
 \qquad R_\lambda(\theta)=D(q_\lambda\Vert Q_\theta).
\tag{C2}
\]

教师改变时两者都使用 q_lambda。根问题是八维

\[
 f_\lambda(a)=F_\lambda(0,a,0),\qquad
 G(a,\lambda)=\nabla_a f_\lambda(a)=\sum_yq_\lambda(y)g_y(a)=0.
\tag{C3}
\]

固定 R01 的数学单位角方向，绝不逐 lambda 重新选择：

\[
 d=u/\sqrt S,\quad
 u=(-8440215,-6758298,-4585241,-1673686,1673686,4585241,6758298,8440215),
 \quad S=281474961915412=\sum_i u_i^2.
\tag{C4}
\]

所以 sum d_i=0、||d||_2=1 且 d 非常量。归一化应保留为精确 u_i u_j/S 等代数式，不用打印后的浮点 d 替代。沿 W_i(t)=a_i(cos(td_i),sin(td_i)) 定义角风险曲率 C(a,q;d)。希望在一个已量化的连通 I 内证明：存在与旧真根相接的 a(lambda)，完整驻点 (0,a(lambda),0) 在16+7方向严格稳定，且 C(a(lambda),q_lambda;d)<0。

在 |lambda|<=1/10 的开发几何范围，第一行与第 j>1 行的角差位于 [-3/2,-1/10]，因此两行内积 rho^2 cos(角差)>0，且其 lambda 导数 -rho^2 sin(角差)>0；其他行对内积不变。该族不是共同旋转。一维 Gaussian logit N(0,rho^2) 及公平观察边际恒定。若要给观察依赖的严格变化，令 kappa(c)=E[tanh(rho U/2)tanh(rho V/2)]，其中 (U,V) 为相关系数 c 的标准 Gaussian；对 Gaussian 密度求导并两次分部积分给

\[
 \kappa'(c)=\rho^2 E[\operatorname{sech}^2(\rho U/2)
 \operatorname{sech}^2(\rho V/2)]/4>0,\qquad |c|<1.
\tag{C5}
\]

故第一项与其余项的观察 spin 成对矩严格变化。C5 只说明改变是真实的，未量化其效应大小，也未认证整个开发范围内的目标稳定性。

## 2. 互补对称仍充分保证 bias 梯度为零

对任意 lambda，由教师零偏置、Gaussian 中心对称及 sigma(-x)=1-sigma(x)，

\[
 A^*_{1-y}(z;\lambda)=A^*_y(-z;\lambda),\qquad
 q_\lambda(1-y)=q_\lambda(y).
\tag{C6}
\]

这不使用反序角度。对候选有逐点关系 A_(1-y)(-z;b,W)=A_y(z;-b,W)。在标签和全部银行粒子同时作 y->1-y、Z_k->-Z_k，得

\[
 F_\lambda(-b,W)=F_\lambda(b,W).
\tag{C7}
\]

这是整个 b 向量同时反号。可微性使每个分量 partial_b_i F_lambda(0,W)=0；再对 W 微分给 H_(b,W)=0。它不声称各个 b_i 能独立反号，也不声称 H_bb=0。

候选第二 Gaussian 坐标反射另给 F_lambda(b,u,-v)=F_lambda(b,u,v)，无论 q 是什么。因此 v=0 时 g_v=0、H_((b,u),v)=0。在 b=v=0：

\[
 \nabla F_\lambda=(0,G(a,\lambda),0),\qquad
 H_{F_\lambda}=\operatorname{diag}(H_{bb},H_{aa},H_\perp).
\tag{C8}
\]

所以八维 C3 的每个根都是完整24维驻点。单条教师角扰动破坏反序对称，**不会**破坏 C6/C7 的零偏置驻定证明。相反，lambda!=0 时不能继续假设 g_a 反序对称，不能仅解原四个 paired 变量，也不能据四维 Hessian 判断完整局部类型。

在 lambda=0，原四维对称根经旧证明已经是八维 G 的根。新八维区间根如何与它相认，见第7节；无需重新求其精确坐标。

## 3. 无新增积分的教师一、二阶变化界

仅第一 Bernoulli 因子随 lambda 变化。写 h=W_(star,1)(lambda) dot Z、h'=W'_(star,1)(lambda) dot Z、h''=-h，B_(y_-1)(Z) 为其余七因子乘积。则

\[
 q'_{\lambda,y}=E[(2y_1-1)\sigma'(h)h' B_{y_{-1}}],
\quad
 q''_{\lambda,y}=E[(2y_1-1)\{\sigma''(h)(h')^2-\sigma'(h)h\}B_{y_{-1}}].
\tag{C9}
\]

sigma'<=1/4、|sigma''|<=1/(6 sqrt3)，h 和 h' 都为 N(0,rho^2)。对固定 Z，有 sum_(y_-1) B=1；因此先用 |E X|<=E|X| 再求和得到全实 lambda 的统一界

\[
 \|q'_\lambda\|_1\le L_1:=\frac\rho2\sqrt{2/\pi}<0.319154,
 \quad
 \|q''_\lambda\|_1\le L_2:=\frac{\rho^2}{3\sqrt3}
 +\frac\rho2\sqrt{2/\pi}<0.442323.
\tag{C10}
\]

上述系数已经包括 y_1 两个取值；没有 256 标签倍数。无需假设 h 与其余教师因子独立。C10 只用单个 Gaussian 一、二阶矩和 sigmoid 导数帽，不调用新教师积分。

因此

\[
 \|q_{\lambda+s}-q_\lambda\|_1\le L_1|s|,\qquad
 \|q_{\lambda+s}-q_\lambda-sq'_\lambda\|_1\le\tfrac12L_2s^2.
\tag{C11}
\]

若有中心严格 q/q' 区间，C11 给整段 Taylor 余项的 **总 L1** 预算；若只给每标签区间，不得把一个总 L1 余项重复当成每标签独立误差后又漏掉标签求和。C9 也可直接在教师积分器内求 q'、q'' 区间。C10 已足以补节点间覆盖，但宽区间若由一处 q 中心外推会非常保守。

本文件的十进制帽由有理平方比较核对：pi>3.1415926、(0.797885)^2*3.1415926>2，(1.73205)^2<3，且 (16/25)/(3*1.73205)+0.319154<0.442323。实际核验只用 Fraction 四则运算；pi 的标准有理下界不是项目积分结果。

## 4. q 变化与 a 变化必须分别入账

固定 a 时，G、完整三块 Hessian 及角风险曲率都对教师 q 线性。令 Z_1,...,Z_K 为候选共线方向的标准 Gaussian，M=max_k|Z_k|。单标签候选行长得分的范数 <=sqrt(m)M，从而

\[
 \|g_y(a)\|_2\le\sqrt{mK}=8.
\tag{C12}
\]

单银行负 log 混合的 Hessian 是 E_pi B-Cov_pi(score)，其中两项都是 PSD。对 (b,a) 块，||B||<=(1+M^2)/4，||Cov||<=m(1+M^2)。PSD 差的 operator 范数不超过两项范数的最大值，所以 E M^2<=K 给

\[
 \|H_{ba,y}(a)\|_{op}\le m(1+K)=72,
 \qquad \|H_{aa,y}(a)\|_{op}\le mK=64.
\tag{C13}
\]

这里是完整单标签 ba Hessian 的界；对具有 C6 的教师加权后 mixed ba 块才精确为零。按条件 Gaussian 第二坐标先积分，单标签横向 Hessian 为

\[
 H_{\perp,y}=E\left[\sum_k\pi_k\operatorname{diag}D_k
 -\sum_k\pi_k(1-\pi_k)r_kr_k^T\right].
\tag{C14}
\]

两项也是 PSD，分别有范数帽1/4和 m sum pi_k(1-pi_k)<=m，因此保守用 ||H_(perp,y)||<=8 即可；K=8 时第二项还可用7，本文不依赖该改进。C13/C14 是由原银行定式得到的新简化包络，不回写旧证书中146.25/8.25的合法但较松预算。

对任意两个教师 PMF q, qbar，令 delta_q=||q-qbar||_1，以上直接给

\[
 \|G(a,q)-G(a,\bar q)\|_2\le8\delta_q,
\quad \|H_{ba}(a,q)-H_{ba}(a,\bar q)\|_{op}\le72\delta_q,
\quad \|H_\perp(a,q)-H_\perp(a,\bar q)\|_{op}\le8\delta_q.
\tag{C15}
\]

固定 q 时，旧全24维 L_F=14069 仍可控制 a 的局部扰动；它对每个 q 都成立。实现中可用更紧的局部区间导数代替该全局帽，但不能未经证明直接减小它。

角风险沿用 R01 的精确标量式。令 p_i=sigma(a_i z)、r_i=y_i-p_i、A_y=product Bernoulli 因子、Q_y=E A_y，

\[
 P_y(z;a,d)=\sum_{i<j}a_i a_j(d_i-d_j)^2r_ir_j,
 \quad N_y(a)=E[A_yP_y],\quad
 C(a,q;d)=\sum_yq_y B_y(a),\quad B_y=N_y/Q_y.
\tag{C16}
\]

在 max |a_i|<=A 的盒内，|B_y|<=m A^2，因为 sum_(i<j)(d_i-d_j)^2=8。因此

\[
 |C(a,q;d)-C(a,\bar q;d)|\le8A^2\delta_q.
\tag{C17}
\]

当 A<=1 时，R01 给固定 q 下 |C(a,q;d)-C(abar,q;d)|<417||a-abar||_2。若预测中心为 c(lambda)，正确的根传递是

\[
 C(a(\lambda),q_\lambda;d)
 \le U_{\rm pred}(\lambda)+417\,\|a(\lambda)-c(\lambda)\|_2,
\tag{C18}
\]

其中 U_pred 必须已经针对 c(lambda) 和 **同一个 q_lambda**。若 U_pred 只针对段中心 q_(lambda_j)，再独立加入 C17 的教师变化预算。预测中心从原 a_0 移动了多少，应由新积分/区间 Taylor 模型实际评价；不对整个 ||c(lambda)-a_0|| 再机械支付417。

## 5. 一段参数上的根包含与可计算接口

取紧区间 I_j、连续且可微的精确预测中心 c_j(lambda)（如 dyadic 系数多项式）、物理 Euclidean 半径 r_j>0，以及固定非奇异8x8预条件矩阵 C_j。定义移动球

\[
 \mathcal B_j(\lambda)=\{a:\|a-c_j(\lambda)\|_2\le r_j\}.
\]

如果严格区间计算同时给出

\[
 \eta_j\ge\sup_{\lambda\in I_j}\|C_jG(c_j(\lambda),\lambda)\|_2,
\quad
 \kappa_j\ge\sup_{\lambda\in I_j,\ a\in\mathcal B_j(\lambda)}
 \|I-C_jH_{aa}(a,\lambda)\|_{op},
\quad \eta_j+\kappa_jr_j<r_j,
\tag{C19}
\]

则每个 lambda 的映射 T_lambda(a)=a-C_jG(a,lambda) 将闭球映入其内部且严格收缩。Banach 定理给球中唯一根，且

\[
 \|a_j(\lambda)-c_j(\lambda)\|_2\le e_j:=\eta_j/(1-\kappa_j)<r_j.
\tag{C20}
\]

严格不等式需包含舍入、积分与所有参数变化误差。C19 本身给 H_aa 非奇异；结合光滑 G 和统一内侧余量，根随 lambda 至少 C1。它不证明球外没有别的根，也不自动证明局部极小。

可选的无预条件存在性接口：若整段球内 H_aa>=mu_j I、mu_j>0，且 sup_lambda||G(c_j(lambda),lambda)||<mu_j r_j，则球面向外与强凸性同样给唯一八维根，误差半径可取 residual/mu_j。球面论证逐 lambda 适用；中心不必固定，不要求反序对称。

实际导数预测的正确公式是

\[
 a'(\lambda)=-H_{aa}(a(\lambda),\lambda)^{-1}
 \sum_yq'_\lambda(y)g_y(a(\lambda)).
\tag{C21}
\]

其中 g_y 与 H_aa 均在随 lambda 变化的根评价。若只使用全局帽和 H_aa>=mu I，则 ||a'||<=8 L_1/mu；二次求导还给保守帽

\[
 \|a''\|\le\{14069\|a'\|^2+128L_1\|a'\|+8L_2\}/\mu.
\tag{C22}
\]

C22 中混合项是2 partial_lambda H_aa a'，使用64 L_1。它可作合法兜底或插值误差估计，通常过松，不应据此锁死可达范围。数值 predictor 本身从来不是根存在证书。

实现可以用候选盒覆盖球：a_i in range(c_(j,i)(I_j))+[-r_j,r_j]。盒比移动球大，但在该盒上得到的 H/导数包络足以用于 C19。若过度包裹，应缩小 lambda 段或保留同一 lambda 变量的 Taylor 相关性；不得把只在离散中心通过的 C19 当作整段通过。

## 6. 固定切片、a 垂直空间与完整稳定性

选择固定非零精确向量 c_star（可取原 a_0），固定切片 c_star^T v=0。令 V 为该超平面的正交列基。在每段根管上核

\[
 \tau_j:=\inf c_*^T a>0,\qquad
 H_{ba}(a,\lambda)\succeq\mu_{16,j}I_{16},\qquad
 V^TH_\perp(a,\lambda)V\succeq\mu_{T,j}I_7,
\quad\mu_{16,j},\mu_{T,j}>0.
\tag{C23}
\]

所有下界须覆盖真正的根；覆盖整段根管是易实现的充分条件。若用有理非正交基 V_r，正确的检验是 V_r^T H_perp V_r-mu_T V_r^T V_r 正定；不能把坐标最小特征值当物理曲率。完整16维块是 diag(H_bb,H_aa)，包含八偏置和八自由行长；不能删去原反对称行长方向。

共同旋转 (u,v)->(u cos alpha-v sin alpha,u sin alpha+v cos alpha) 在共线点的切片约束导数为 c_star^T a。C23 的统一正下界使切片沿整段横截。C8 加 C23 给23维切片 Hessian 正定，由旋转不变性得到每个根都是模共同旋转的严格局部极小。

Ward 恒等式在共线点为 H_perp a=G(a,lambda)，只有在**真根**上才能用 H_perp a=0。令 gamma=|c_star^T a|/(||c_star|| ||a||)>0。在真根上，对 x in c_star-perp，

\[
 x^T H_\perp x=(P_{a^\perp}x)^TH_\perp(P_{a^\perp}x),
 \quad\|P_{a^\perp}x\|\ge\gamma\|x\|.
\tag{C24}
\]

因而若动态空间 a-perp 的物理曲率>=mu，则固定切片曲率>=mu gamma^2。反向地，对 w in a-perp 取 x=w-a(c_star^T w)/(c_star^T a)，则 x in c_star-perp、x^THx=w^THw 且 ||x||>=||w||；固定切片曲率>=mu_T 推出 a-perp 曲率>=mu_T。二者由 C23 的统一横截性联系，不能在非根处忽略 H_perp a=G 的残差后直接互换。

若另外核 a_i>=a_min>0，则固定方向的角切向量 w=a odot d 满足

\[
 \|P_{a^\perp}w\|^2=\min_c\sum_i a_i^2(d_i-c)^2\ge a_{\min}^2,
\tag{C25}
\]

使用了 sum d_i=0、||d||=1。因此在真根上角路径 F_lambda''(0)>=mu_T a_min^2>0；路径加速度与完整梯度的内积为零。风险的路径加速度则已包含在 C16 中，不能用 Cartesian 风险横向 Hessian 二次型替代 C16。

## 7. 相邻参数段及旧真根的唯一连接

逐段存在唯一根不自动意味着各段属于同一分支。只验证相邻根球相交也不够：相交域中未必含有任一根。对相邻段公共端点 lambda_s，以下任一接口充分：

1. 从第 j 段取得包含其真根的收紧 enclosure E_j(lambda_s)，验证 E_j(lambda_s) 完全包含于第 j+1 段的唯一根球；于是该根就是后段根。
2. 认证一个共同凸域内 H_aa>=mu I>0，并验证两个端点根 enclosure 都落在该域；强单调性使两根相等。
3. 单独作覆盖两侧候选的端点区间根证书，并证明两侧根 enclosure 都进入其唯一性域。

连接处 Hessian 非奇异和局部隐函数唯一性使两侧 C1 分支相接；逐段证据给整个连通 I 的同一条分支。若两个段有区间重叠，不能仅比较各自中心读数；仍须在至少一个共同参数点证明同根，并用局部唯一延续连接重叠部分。

lambda=0 的起始识别用原真根的公开证明，不要另选同教师下一个数值解冒名。旧根满足 ||a_old-a_0||<1.805e-8；旧 H_aa 在原物理球 ||a-a_0||<=5e-7 上已有严格正下界（原全16维界的 principal block），故该球内其实至多有一个**八维** G 根。可以验证新的 lambda=0 根 enclosure 在此旧球内，或验证旧根的1.805e-8包围完全进入新收缩球。任一路径都将新八维根与旧四维对称构造的同一个 a_old 连接。仅比较近似根的小数不够。

## 8. 直接区间积分怎样覆盖参数和候选盒

各段需要两类独立输入：教师 q_lambda 的严格区间/Taylor enclosure，和候选 a 盒上的单标签函数 enclosure。不得把新教师 q 当作候选 Q，也不得随 root 漂移沿用原 paired kernel 而未说明适用性。

在一般八载荷共线点，令 z 为一维标准 Gaussian，u=exp(v)，

\[
 \Psi_y=E e^{-uA_y},\quad t_y=uA_y e^{-uA_y},\quad
 T_0=E t_y,\quad M_0=E[t_y r],\quad M_1=E[t_y z r],
\]
\[
 E_0=E[t_y rr^T],\quad E_2=E[t_y z^2rr^T],\quad
 D_0=E[t_y\operatorname{diag}(p_i(1-p_i))],\quad
 D_2=E[t_yz^2\operatorname{diag}(p_i(1-p_i))].
\]

先对完整坐标求导再在共线点评价，原 Laplace 恒等式给

\[
 g_y=-\int K\Psi_y^{K-1}M_1\,dv,
\]
\[
 \begin{aligned}
 H_{bb,y}&=\int\{K\Psi_y^{K-1}D_0-K(K-1)\Psi_y^{K-2}(T_0E_0-M_0M_0^T)\}\,dv,\\
 H_{aa,y}&=\int\{K\Psi_y^{K-1}D_2-K(K-1)\Psi_y^{K-2}(T_0E_2-M_1M_1^T)\}\,dv,\\
 H_{\perp,y}&=\int\{K\Psi_y^{K-1}D_0-K(K-1)\Psi_y^{K-2}T_0E_0\}\,dv.
 \end{aligned}
\tag{C26}
\]

C26 对任意 a 成立，不依赖行长配对。负号、same/distinct 粒子项以及两个因子的共享标签必须保留。教师 q 只在单标签函数形成后线性加权。互补标签 y,1-y 在 z 反射后具有相同 g_a、H_bb、H_aa、H_perp、Q_y 和 B_y，可以安全使用128个互补代表；**一般八载荷不再支持旧41个 paired-count代表**。若在恰好 paired 的预测点复用41类，仍须保留原文的 W/M/S 标签矩，且不能把这个特例推广到覆盖非paired a 的整个盒。

一种充分且可核验的实现流程是：

1. 对每个有限 z/v 节点，把 lambda 和 a 的允许范围作为严格实区间或带相关性的 Taylor 模型输入，得到真实有限和的 enclosure。教师也可在固定 lambda 中心计算 q/q' 严格区间，再用 C11 覆盖整段。
2. 将候选 Gaussian 尾、Laplace 离散上下尾、复条带梯形误差及内层误差传播证明，统一扩到本段候选盒。至少使用 C_a=sup sum|a_i|、A_max=sup max|a_i| 检查条带相位 C_a d_z<pi/2、分母正性、外尾阈值等。原四维缩减坐标常数须改成八维物理向量常数；不能直接复制原某中心的数值误差帽。
3. 风险用 C16 的 Q_y 和 N_y 严格一维积分，并验证 Q_y 下端点>0 后区间相除。它们在候选盒上的包络可直接覆盖根，也可只评价预测中心，再以 C18 传到半径 e_j 内。前者若包裹过宽，后者通常更便宜。
4. 加入所有算术区间、教师包围及求积误差，形成 C19/C23 和风险上端点。若每标签有统一候选误差 epsilon，真实 PMF 加权仍只有 epsilon；若先使用近似非负教师权重 q_f，则严格使用质量上界 M_f*epsilon 加 C15/C17 的总 L1 误差。不可因 midpoint 重归一偷偷改变 q_f，也不可把256或128重复乘到一个已经是标签一致且按概率平均的误差界。
5. 用带 Gram 的严格 LDL、interval residual/operator 界等验证物理谱与收缩；逐元素区间应通过 Frobenius 或其他严谨 operator 界转换。浮点特征值、粗细格差、有限节点扫描只能是开发读数，不能代替这些不等式。

如需保证所有 q 约束，独立坐标盒允许 sum q 的区间宽于1并不影响包含性，但可能很松；可保留 sum q=1、q>=0、互补等线性约束做更紧的加权包围。该数值改进不能把未经证实的区间相关性当作等式。

完成所有段的 C19/C23、连接证据以及统一风险上端点 <=-nu<0 后，才可宣称整个 I 的持续性。由紧区间上的光滑性、统一正/负二阶余量及 a_min>0，可进一步得到一个共同充分小 t_0>0：所有 lambda in I、0<|t|<t_0 下，F_lambda 上升、R_lambda 下降，且角路径行长、logit边际、观察边际保持，载荷秩为二。若未计算相应高阶余项，只给 t_0 的存在，不报未认证的有限步长或效果量。

## 9. 旧全局常数为什么只能给很小的保守兜底

以下只是对一种**固定中心证明方法**的预算说明，不是实际分支位移或最大持续范围的上界。使用旧安全帽 L_F=14069、原 r=5e-7、旧矩阵距离帽.001，得 m_16=.0419655、m_T=.0059655。以 h=sup|lambda|，C10/C15 给固定 a_0 的残差帽

\[
 \epsilon(h)=7.572\times10^{-10}+8(0.319154)h,
\quad m_{16}(h)=.0419655-72(0.319154)h,
\quad m_T(h)=.0059655-8(0.319154)h.
\tag{C27}
\]

只用 epsilon(h)<r m_16(h) 得 h<7.9215124146e-9；在这个尺度 m_T 仍正。这说明原球加统一教师 L1 帽确能作为极小起始连接预算，却不足以成为有意义的 R03 终点。甚至忽略其他误差，单用旧横向.014与14069，允许的 a 漂移量级也只有 .014/14069≈9.9509560e-7。

同样，R01 根曲率负余量.00450377如果全部用417偿付，允许 a 位移仅约1.0800408e-5；若只用 C17 的 A=1 教师帽，单独的教师变化成本已经是2.553232 h，耗尽该余量的尺度约1.76395e-3。三个数都只是这些粗界的局限，不能当真实曲率过零点、真实分支界或不可能性证明。

实际可达范围应依赖沿 a(lambda) 的新候选评价、局部误差小管和逐段连接。可使用417支付很小的 e_j，使用14069支付很小的根包围余量；不把总 branch drift 作为同一个误差半径。若最终严格范围仍小，必须如实报告教师依赖变化量和失败预算，不能仅把“正半径”包装成主要新结果。

## 10. 本文件的执行与剩余前提

本次实际执行只做 Fraction 标量四则运算：C10 的十进制上帽、C27 的固定球阈值、.014/14069 和 .00450377/417 等预算读数。没有真实目标积分、优化、教师 PMF 生产、科学数组读取或新科研结论。C6–C8 是解析对称性证明；C19–C26 是后续生产可直接填数的接口。

尚待实际证据的是各段 q 与候选函数包围、根收缩、完整谱正性、固定角风险负号、旧根识别、段间唯一连接，以及认证区间内教师/观察依赖的量化变化。任一门不通过都须保留失败或未分辨状态；不得用开发节点上同号代替连续区间证书。
