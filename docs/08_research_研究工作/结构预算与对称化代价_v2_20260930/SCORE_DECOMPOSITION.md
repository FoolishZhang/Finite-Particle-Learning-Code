# 完整模式评分中的模型代价和有限计算代价

本文件解释本轮v2实验的对象，并给出高预算复查的解析依据。它不回写v1。分解来自交叉熵恒等式与Jensen不等式，收敛来自标准大数律与一致可积性；不宣称这些一般原理为新定理。

## 有限粒子评分的两个非负项

固定教师P、候选Q和明确群平均G。令正似然估计器的均值为Q_G=S_GQ，bank独立于当前标签，并定义

\[
F_{K,G}=E_{C\sim P,\omega}[-\log\widehat Q_{K,G}(C;\omega)],
\quad J_{K,G}=E_P[\log Q_G(C)-E_\omega\log\widehat Q_{K,G}(C;\omega)]\ge0.
\]

在本有限状态、正概率、有限期望的设置中，

\[
F_{K,G}-H(P)=D(P\Vert Q_G)+J_{K,G}.
\tag{1}
\]

J是对数估计的Jensen gap，不是方差；只有额外的渐近条件下才可用相对方差近似它。离开真参数时，S_GQ可能反而更接近P，因此不能普遍说群平均必然恶化风险。

在本轮真参数处，正确group-count的均值为P；错误single-count的均值为S(P)。记D_sym=D(P||S(P))，则

\[
F_{K,S}-F_{K,G}=D_{\rm sym}+J_{K,S}-J_{K,G}.
\tag{2}
\]

因此，错模型可以取得更低的有限评分，恰当且有限的解释是其Jensen gap下降超过了模型代价。模型项、评分项和残差必须使用同一个完整模式单位；不能把count NLL替代模式NLL，也不能把三项代数相关量当三份独立证据。

## 总计数评价为何会精确遗漏模型差

给定同一latent bank，raw、组内平均和全置换平均都保持每个粒子的总count分布。故对每个bank、每个n，

\[
\sum_{|a|=n}\widehat Q_{K,\rm raw}(a)
=\sum_{|a|=n}\widehat Q_{K,G}(a)
=\sum_{|a|=n}\widehat Q_{K,S}(a).
\tag{3}
\]

所以基于总count的风险在相同bank下完全相同，不只是统计上“未显著不同”。R2/C使用独立Poisson-binomial表达和对角求和核对此恒等式。

raw的canonical模式计算只在对组内不变的bank取期望后，才等于完整教师风险。其canonical数组不被当作归一化的固定bank观察分布；式(3)依赖原完整product mixture的count恒等式。

## 相同count下的更直接判断式

以下在R2结果出现后作解析整理，并在C既定输出上检验，不改变任一原主门。写p为真实group-count联合PMF，q_ω为某个bank的正确group-count PMF，p_N、q_{ω,N}为各自的总count边际。令h(a,b|n)=choose(m,a)choose(m,b)/choose(d,n)为全置换模型的条件组分配。定义

\[
i_\omega=D(p_N\Vert q_{\omega,N}),\qquad
a_\omega=\sum_n p_N(n)D\big(p(\cdot\mid n)\Vert q_\omega(\cdot\mid n)\big).
\]

定义L_G(ω)=E_{C~P}[−log qhat_G(C;ω)]，L_S同理；它们是固定bank的完整教师加权风险，不是单个观察的loss。因为P及group评分在每个组count内都均匀，完整模式KL与group-count KL相同。KL链式分解逐bank给出

\[
L_G(\omega)-H(P)=i_\omega+a_\omega,\qquad
L_S(\omega)-H(P)=D_{\rm sym}+i_\omega,
\]

\[
\boxed{L_S(\omega)-L_G(\omega)=D_{\rm sym}-a_\omega.}
\tag{3a}
\]

其中D_sym=Σ_n p_N(n)D(p(·|n)||h(·|n))。所以错single有限评分更好，当且仅当该bank的组分配近似误差a_ω超过错误对称化的代价；取bank期望后的判据是E a_ω>D_sym，不是多数bank偏好。总count项在比较中精确抵消。

这比泛称“方差减少掩盖偏差”更具体：被比较的是同一总count边际下的条件分配误差。i_ω与a_ω都是非负KL，且single的Jensen gap正好是E i_ω。a_ω不是从coarse事件下界得到的量，也不自动等于统计方差。

这些量依赖已知教师参考，只用于受控研究，不是未知自然分布上免费可用的选择准则。C保存完整逐bank PMF，可直接重算i_ω和a_ω；R2保存的风险和count风险只用于代数读回。它们与原配对主差是同一条证据，不增加独立成功次数。

## 固定维数下排名为何最终必须恢复

以下是R2完成后写明的解析解释及C包依据，不倒称为R原预注册结论。固定偶数d≥4、两个不等的有限正尺度及本包两个已知等大组；采用iid先验粒子。对每个模式a，group与single的单粒子似然W(a)均在(0,1]，且

\[
-\log W(a)\le d\log2+\|\eta\|_1+\log M_a,
\]

其中M_a为相应orbit大小。因为平均中含原模式的似然项，且-log sigmoid(t)≤log2+|t|，该界成立；Gaussian输入使右侧期望有限。

设Z_i=-log W_i，A_K=K^{-1}Σ_i Z_i，X_K=-log(K^{-1}Σ_i W_i)。Jensen给0≤X_K≤A_K。对任意L>0，将Z_i分为≤L/2和>L/2两部分，由Markov不等式可得

\[
E[A_K\mathbf1_{A_K>L}]
\le2E[Z_1\mathbf1_{Z_1>L/2}]\longrightarrow0
\]

且对K一致。因此X_K一致可积。大数律给X_K→-log EW，进而期望也收敛；再对有限模式集求和，得到

\[
F_{K,G}\longrightarrow H(P),\qquad
F_{K,S}\longrightarrow H(P)+D_{\rm sym},
\]

\[
F_{K,S}-F_{K,G}\longrightarrow D_{\rm sym}>0.
\tag{4}
\]

所以，若有限预算下错single方法评分更低，这种偏好不可能在K趋无穷时一直维持。式(4)不保证单调、唯一交叉点或某个给定K已经足够，也不自动给高概率的逐bank结果。C包独立冻结所测试的两个高预算，不按结果继续找交叉。

null profile下P本来全交换，D_sym=0。全置换平均是更粗orbit上的条件期望，故single的总体评分不劣于group，并趋向相同精确风险；它不是错模型对照的正例。

## 数值资格与统计范围

完整模式P及S(P)的高精度参考由R2独立Fourier积分给出。其复数因子不是正权product分量，因此参考求值不违反有限product-mixture界。三种参考数值的一致性、独立Hermite矩、解析尾界和误差估计分别记录，不冒称完整舍入/求积误差的严格证书。

有限bank的t区间条件于这份参考；参考变体同时回放到H、模型KL、F权重、S−G以及J残差，其敏感性另列。原R失败保留，不因R2修复而追认为通过。R2、E和C的原置信家族分别解释；联合敏感性若报告，必须注明实际alpha分配与其事后开发身份。
