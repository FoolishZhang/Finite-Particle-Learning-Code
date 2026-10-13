# 附录 D. 固定有限搜索的新运行风险

## D.1. 数学程序及其风险分离

固定中心化三变量 Gaussian-logistic 模型、独立教师 $C_*=I$、单位边际方差、$K=100$ 及 $n=256$。候选集合 $\mathcal C_{\rm grid}$ 包含 $\{-.9,-.6,-.3,0,.3,.6,.9\}^3$ 中使单位对角相关矩阵满足 $C\succeq.1I$ 的143个三元组。真值 $I$ 和非对角元均为 $.9$ 的等相关锚点 $C_b$ 都在该集合中。下文均使用正定对称平方根。

每次新运行从 $\{-1,1\}^3$ 的均匀分布抽取 $N=65,536$ 个 iid 活跃标签，另独立抽取 $B=16,384$ 个 iid 银行。银行 $W_b$ 包含 $K$ 个标准三维活跃 Gaussian 向量，每个粒子另有256个独立标准 Gaussian 无关坐标。所有观测和候选共享同一批银行。定义

$$
\begin{aligned}
S_k&=\sum_{j=1}^{256}\log\sigma(G_{kj}),\qquad
a_k(W)=\frac{e^{S_k}}{\sum_{l=1}^{K}e^{S_l}},\\
\phi(C;y,W)&=-\log\sum_{k=1}^{K}a_k(W)
 \prod_{j=1}^{3}\sigma\!\left(y_j(C^{1/2}Z_k)_j\right),\\
\widehat F(C)&=\frac1{NB}\sum_{\ell=1}^{N}\sum_{b=1}^{B}\phi(C;Y_\ell,W_b),\\
d(y,w)&=\phi(I;y,w)-\phi(C_b;y,w),\quad
g=\mathbb E d(Y,W),\quad
\widehat g=\frac1{NB}\sum_{\ell,b}d(Y_\ell,W_b).
\end{aligned}
\tag{D.1}
$$

数学程序返回与最小数学评分相差不超过 $\mathrm{tol}=10^{-12}$ 的词典序首个候选，不使用观察风险比较或回退规则。这个定义将数学求值与其浮点实现区分开来。

每个非独立网格点至少有一项 $|c_{ij}|\ge.3$。对观测对 spin 矩 $\kappa$，附录C已证明 $\kappa(0)=0$ 及 $\kappa'(c)\ge\beta=e^{-1/2}/4$。有界的观测对 spin 统计量使全变差至少为 $|\kappa(c_{ij})|/2$；再沿真值到候选的 KL 方向使用 Pinsker，得到

$$
\begin{aligned}
\mathcal R(C)&\ge\tfrac12\beta^2(.3)^2
=r_{\rm grid}:=\frac{0.09\,\exp(-1)}{32}>.00103466
 \quad(C\in\mathcal C_{\rm grid}\setminus\{I\}),\\
\widehat g>\mathrm{tol}
&\ \Longrightarrow\ 
\widehat F(I)>\min_{C\in\mathcal C_{\rm grid}}\widehat F(C)+\mathrm{tol}
\ \Longrightarrow\ \mathcal R(\widehat C)\ge r_{\rm grid}.
\end{aligned}
\tag{D.2}
$$

第一个蕴含使用 $\min_C\widehat F(C)\le\widehat F(C_b)$，不需要判定其余142点中谁胜出。全文使用精确风险常数；向上舍入的近似值 $.00103466093$ 不是已经认证的风险下界门。

## D.2. 共享银行方差与 Gaussian 尾界

令 $m(y)=\mathbb E_Wd(y,W)$、$q(w)=\mathbb E_Yd(Y,w)$。交叉平均的分解为

$$
\begin{aligned}
d(y,w)&=g+a(y)+b(w)+r(y,w),\qquad
a(y)=m(y)-g,\quad b(w)=q(w)-g,\\
\mathbb E[r(Y,W)\mid Y]&=\mathbb E[r(Y,W)\mid W]=0,\\
\operatorname{Var}(\widehat g)
&=\frac{\sigma_Y^2}{N}+\frac{\sigma_W^2}{B}+\frac{\sigma_r^2}{NB},\\
\sigma_Y^2&=\mathbb E a(Y)^2,\qquad
\sigma_W^2=\mathbb E b(W)^2,\qquad
\sigma_r^2=\mathbb E r(Y,W)^2.
\end{aligned}
\tag{D.3}
$$

条件中心化使残差协方差在任一索引不同时消失，三项之间也相互正交。这证明了 (D.3)，没有将 $NB$ 个交叉项当作独立观测。

坐标置换保持 $C_b$ 及其平方根不变，全局符号反射保持 Gaussian 银行分布不变。因此 $m(y)$ 只有两个取值：两个全同号标签对应 $m_{\rm same}$，其余六个标签对应 $m_{\rm split}$。于是

$$
g=\tfrac14m_{\rm same}+\tfrac34m_{\rm split},\qquad
\Delta=m_{\rm same}-m_{\rm split},\qquad
\sigma_Y^2=\tfrac3{16}\Delta^2.
\tag{D.4}
$$

这只是条件期望的对称性。同一银行实际得到的八个点差不必只有两个值，程序也不对输入作对称化。

$\log\sigma$ 是1-Lipschitz函数，log-sum-exp 关于最大范数同样是1-Lipschitz。两个锚点使用完全相同的无关权重，因此下面的包络不依赖这些权重。附录C取 $t=3/8$ 的 Gaussian 指数矩论证给出

$$
\begin{aligned}
A&=\|\sqrt{C_b}-I\|_{\rm op}
=\max(1-\sqrt{.1},\sqrt{2.8}-1),\qquad
M_Z=\max_{k\le K}\|Z_k\|_2,\\
|d(y,W)|&\le\sqrt3\,A M_Z,\qquad
\mathbb E M_Z^2\le b_K:=\tfrac83(\log K+3\log2),\\
\sigma_r^2&\le\mathbb E d(Y,W)^2\le3A^2b_K.
\end{aligned}
\tag{D.5}
$$

最后一个界来自 (D.3) 的正交性，直接控制残差项，不需要再增加随机审计目标。

取 $D=3K=300$、$T=6$，以及 $E_T=\{\max_{k,j}|Z_{kj}|\le T\}$。定义银行的类别差 $u(w)$ 为两个同号标签点差的平均，减去六个异号标签点差的平均；于是 $\mathbb Eu(W)=\Delta$。三个截断审计变量及其范围为

$$
\begin{gathered}
X=q(W)\mathbf1_{E_T},\qquad
D_T=u(W)\mathbf1_{E_T},\qquad M=3AT,\\
X\in[-M,M],\qquad X^2\in[0,M^2],\qquad D_T\in[-2M,2M].
\end{gathered}
\tag{D.6}
$$

以 $\varphi$ 和 $\overline\Phi$ 表示标准 Gaussian 密度和上尾概率。坐标联合界与 Mills 不等式控制尾概率。为控制二阶矩，用全部 $D$ 个坐标的平方和上界 $M_Z^2$，再用坐标越界指示之和上界尾事件指示。对角项使用 $\mathbb E[Z^2\mathbf1_{\{|Z|>T\}}]=2(T\varphi(T)+\overline\Phi(T))$，非对角项使用独立性；没有假设越界事件互斥。因此

$$
\begin{aligned}
\mathbb P(E_T^c)&\le p_T:=2D\varphi(T)/T,\\
\mathbb E[M_Z^2\mathbf1_{E_T^c}]
&\le2D\{T\varphi(T)+D\overline\Phi(T)\}
\le Q_T:=2D\varphi(T)(T+D/T),\\
\tau_2&:=3A^2Q_T,\qquad \tau_1:=\sqrt{\tau_2p_T},\\
|g-\mathbb EX|&\le\tau_1,\qquad
|\Delta-\mathbb ED_T|\le2\tau_1,\qquad
\sigma_W^2\le\mathbb E q(W)^2\le\mathbb EX^2+\tau_2.
\end{aligned}
\tag{D.7}
$$

Cauchy--Schwarz 给出均值尾界。数值上 $\tau_1\simeq1.31901920513\times10^{-5}$、$\tau_2\simeq2.86347137776\times10^{-4}$；实际计算对这些常数作向外外包。

## D.3. 使用另行生成银行的三分量矩审计

这里是作者使用另行生成银行样本的计算，不是第三方审计。审计使用 $L=262,144$ 个新银行，每个银行内评价全部八标签与两个锚点。这些样本与未来程序的 $N$ 个标签、$B$ 个银行分开。此前的独立银行只用于开发和选择截断设计，没有进入新置信统计。正式运行在生成新银行前固定 $T=6$、三个目标和误差分配。可再生实现采用 PCG64DXSM，种子序列为 $[202610060501,j]$，活跃与无关部分使用独立的 $j=1,2$ 流。概率解释以规定的 iid Gaussian 模型为前提，可再生性本身不证明这一前提。

对每条保存的 binary64 标量序列 $\widetilde V_1,\ldots,\widetilde V_L$，选择共同的二进制分母，并精确累积整数：

$$
\widetilde V_b=2^{-e}z_b,\quad S_1=\sum_bz_b,\quad S_2=\sum_bz_b^2,\qquad
\overline{\widetilde V}=\frac{S_1}{L2^e},\quad
\widetilde s_V^2=\frac{LS_2-S_1^2}{L(L-1)2^{2e}}.
\tag{D.8}
$$

$X$、$X^2$、$D_T$ 的指数分别为56、95、68。之后用高精度球算术外包标准差、置信端点和最终概率。表中的样本矩是舍入后的描述值，不是真实方差的界。

|审计变量|样本均值|无偏样本方差|
|---|---:|---:|
|$X$|.00562444984697731|.04106054244035259|
|$X^2$|.04109202024289478|.011140404592328061|
|$D_T$|.23838858527527407|.05025621128454873|

没有银行触发截断。三项各分配 $\alpha_j=.01/3=1/300$：对 $X$ 的均值作单侧下界，对 $X^2$ 的均值作单侧上界，对 $D_T$ 的均值作双侧区间。对范围长度 $R_j$，将数值样本误差同时传播到均值和标准差，经验 Bernstein 不等式 [@maurer2009, Theorem 4] 使用

$$
\begin{aligned}
\epsilon_X&=10^{-9}+10^{-12},\qquad
\epsilon_D=2\cdot10^{-9}+10^{-12},\\
\epsilon_2&=(2M+\epsilon_X)\epsilon_X+10^{-12},\\
B_j&=\sqrt{\frac{2\ell_j}{L}}\!
 \left(\widetilde s_j+\sqrt{\frac{L}{L-1}}\epsilon_j\right)
 +\frac{7R_j\ell_j}{3(L-1)},\\
(R_X,R_2,R_D)&=(2M,M^2,4M),\qquad
(\ell_X,\ell_2,\ell_D)=(\log600,\log600,\log1200).
\end{aligned}
\tag{D.9}
$$

这里 $|\widetilde V_b-V_b|\le\epsilon_j$ 是相应数值条件。中心化 Euclidean 范数给出 $|s_j-\widetilde s_j|\le\sqrt{L/(L-1)}\epsilon_j$。因此 (D.9) 使用精确数学范围，放宽样本标准差，并在均值端点另加入 $\epsilon_j$。附录E给出完整算术方法和仍保留的输入资格。特别地，后续对全部2,097,152个保存的银行×标签点差进行了外包；当保存的 $Z/S$ 视为精确数时，最大算术误差低于 $3.331\times10^{-14}$，原 $10^{-9}$ 余量保持不变。这次确定性重算不引入新置信家族，也不认证输入生成或所有未来运行。

在依相应条件具有至少 $.99$ 概率的同时审计事件上，向外端点定义为

$$
\begin{aligned}
g_L&=\overline{\widetilde X}-\epsilon_X-B_X-\tau_1,\\
U_2&=\overline{\widetilde{X^2}}+\epsilon_2+B_2+\tau_2,\\
\Delta_-&=\overline{\widetilde D_T}-\epsilon_D-B_D-2\tau_1,\qquad
\Delta_+=\overline{\widetilde D_T}+\epsilon_D+B_D+2\tau_1,\\
U_\Delta&=\max(|\Delta_-|,|\Delta_+|),\\
g&\ge g_L,\quad \sigma_W^2\le U_2,\quad |\Delta|\le U_\Delta,\qquad
\operatorname{Var}(\widehat g)\le V_U:=
\frac{3U_\Delta^2}{16N}+\frac{U_2}{B}+\frac{3A^2b_K}{NB}.
\end{aligned}
\tag{D.10}
$$

银行项使用总体二阶矩上界，不以没有资格的样本方差替代真实方差。下表保守显示认证端点；实际推断使用完整的向外端点。

|量|保守显示值|
|---|---:|
|$g_L$|.00279405023657203|
|$U_2$|.05074113097360336|
|$U_\Delta$|.24317070176659291|
|$V_U$|$3.289456534426161\times10^{-6}$|
|$p_L$|.70355036|

下表给出 $V_U$ 的三项贡献，均向上显示。它们是该方差**上界**的组成，不是经验方差分解或因果归因。

|上界组成|向上显示值|
|---|---:|
|数据：$3U_\Delta^2/(16N)$|$1.691779809884757\times10^{-7}$|
|银行：$U_2/B$|$3.096992857275596\times10^{-6}$|
|残差：$3A^2b_K/(NB)$|$2.328569616208936\times10^{-8}$|

## D.4. 概率陈述及其边界

由于 $g_L>\mathrm{tol}$，对新运行交叉平均使用单侧 Cantelli，再结合 (D.2)，在同一审计事件上有

$$
\begin{aligned}
\mathbb P_{\rm fresh}\{\mathcal R(\widehat C)\ge r_{\rm grid}\}
&\ge\mathbb P_{\rm fresh}\{\widehat g>\mathrm{tol}\}\\
&\ge\frac{(g_L-\mathrm{tol})^2}{V_U+(g_L-\mathrm{tol})^2}
\ge p_L>.7035.
\end{aligned}
\tag{D.11}
$$

具体地，Cantelli 将 $\mathbb P\{\widehat g\le\mathrm{tol}\}$ 上界为 $V/(V+(g-\mathrm{tol})^2)$；该式随 $V$ 增加、随 $g>\mathrm{tol}$ 减小。记录的下端点约为 $.703550362875$，表格和正文采用保守向下的显示值。

**99%是对新运行风险概率下界的审计置信度。** 70.35%是一个下界，不是观察到的失败频率。本审计没有反复执行143点搜索，也没有抽取新的训练标签；它建立的是这一充分条件论证所需的矩界。KL 的非负性还给出确定性推论

$$
\mathbb E_{\rm fresh}\mathcal R(\widehat C)
\ge .7035\,r_{\rm grid}>.00072788.
\tag{D.12}
$$

它使用同一审计事件，不花费额外误差预算。

结果只针对 $n=256$ 程序本身，没有证明至少70.35%的匹配新运行会从正确的 $n=0$ 输出变成错误的 $n=256$ 输出，也不主张每个错误输出都等于 $C_b$ 或具有 $.0311$ 风险。更细的网格可能含有其他低风险点，使单锚点捷径失效。附录C的连续区域结论与本附录的粗网格概率结论提供不同保证；二者都不是 Adam 或任意连续优化的保证。

浮点程序需要另一个条件。假设在新运行概率至少为 $1-\delta_{\rm alg}$ 的事件上，所有候选评分都有限、程序按规定容差返回可行网格点，并且 $|\widehat g_{\rm fp}-\widehat g|\le\epsilon_{\rm alg}$。令 $t_{\rm alg}=\mathrm{tol}+\epsilon_{\rm alg}<g_L$，由联合界得到

$$
\mathbb P_{\rm fresh}\{\mathcal R(\widehat C_{\rm fp})\ge r_{\rm grid}\}
\ge\left[
\frac{(g_L-t_{\rm alg})^2}{V_U+(g_L-t_{\rm alg})^2}
-\delta_{\rm alg}\right]_+,\qquad [x]_+=\max(x,0).
\tag{D.13}
$$

左侧事件只计入满足风险门的有效网格输出；未成功返回的执行不计入。 如果有效输出与误差保证一致成立（$\delta_{\rm alg}=0$），$\epsilon_{\rm alg}\le.00022$ 可保留超过 $.66824>2/3$ 的概率下界。这只是条件性敏感性陈述：当前没有建立这样的统一保证。保存输入上的算术外包不能认证 Gaussian 生成、无关变换的累积、理想 iid 抽样律或上述未来程序条件。

本审计是新增的99%家族。将它与附录C、F中的锚点差、中心曲率和已知均值三个家族合并，保守联合覆盖至少为96%；再加入附录H较早的有限选择家族，则至少为95%。这些界只覆盖所列家族，不是论文所有历史实验的整体覆盖率，也不要求家族之间独立。全部保存输入的算术加强和 (D.12) 都不增加统计比较。

\newpage

## D.5. 同一审计事件上的连续区域推论

式 (D.1) 中的 $g$ 正是附录 C 的 $F_{256}(I)-F_{256}(C_b)$。两种定义都将相同的归一权重损失对均匀三位教师标签和同一 Gaussian 银行律取平均，使用已知零均值、单位边际方差、$K=100$、$n=256$ 和相关为 $.9$ 的等相关锚点。与候选无关的无关权重归一化常数在点差及总体 regret 中抵消。$T=6$ 只截断审计统计量，不改变总体目标：式 (D.10) 中的均值尾修正已使 $g_L$ 成为未截断 $g$ 的下界。

式 (C.4)–(C.5) 的半径与总体代价 $c_d=H(b_{100};1-d)d^2/2$ 都是确定性的，不需要额外随机事件。在本节矩审计的同一99%联合事件上，保留其输入、数值与 iid 模型资格，应用该区域不等式得到

$$
\begin{aligned}
\inf_{\substack{C\in\mathcal E_{.1}^{(3)}\\\mathcal R(C)\le10^{-6}}}
\{F_{256}(C)-F_{256}(C_b)\}
&\ge g_L-c_d>.00134,\\
F_{256}(C)-\inf_{C'\in\mathcal E_{.1}^{(3)}}F_{256}(C')<.00134
&\ \Longrightarrow\ \mathcal R(C)>10^{-6}.
\end{aligned}
\tag{D.14}
$$

向外标量算术用已有 $g$ 的下端点和 $c_d$ 的上端点核实上述严格余量。第二行来自总体下确界不大于 $F_{256}(C_b)$。因此，支持新运行概率陈述的审计事件，也支持连续区域蕴涵（两者完整陈述见C.15），其保守 regret 门为 $.00134$。

这是同一事件的确定性推论，没有新增统计比较或alpha分配，也没有给出连续优化器的新运行失败概率。附录 C 中原来约 $.0008952$ 的余量和最初七门六过的记录保持不变。四个或五个指定家族的联合覆盖仍分别为96%和95%，本推论不改变其范围。
