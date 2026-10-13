# 全局计数结构与有限MC门槛：独立理论审查

日期2026-09-29。本文件只作解析推导与定向文献核对；没有训练、模拟、数值积分或新评价结果。它不改写旧冻结，也不认证原创性。下文使用前轮已独立核验的局部公式，但把真分布换成新的明确族。

## 1. 共同训练目标和局部结论

令真二值向量为C，正确边际为p，mu0=logit(p)，D=diag[p_j(1-p_j)]，Gamma=Cov(C)。考察自由输出因子B的Gaussian-logistic模型，eta=mu+BZ，以及K个iid先验样本的期望负log-mean-likelihood。独立点(mu0,0)的Hessian在均值/因子方向分块为

\[
 \operatorname{diag}\{D,I_q\otimes A_K\},\qquad
 A_K=D-(1-1/K)\Gamma.
\]

因此K>1时，独立点严格局部极小的充分且在非临界情形必要条件为

\[
 \lambda_{\max}(D^{-1/2}\Gamma D^{-1/2})<K/(K-1).
\]

严格反向不等式给鞍点；相等时需要更高阶。K=1时因子Hessian为D。对标准Z/-Z配对，A_anti=D-Gamma；正确边际、非零两两协方差使它非零且迹为零，故独立点是鞍点。这些是每个固定d的输出参数空间陈述，不是关于神经权重空间、有限步训练或全局最优的结论。前轮证明见局部极小推论 (historical reference not distributed)。

## 2. 精确m-of-d：公式正确，但不能冒充正确指定Gaussian族

设d>=2，1<=m<=d-1，P在所有总和为m的二值向量上均匀。令p=m/d，则

\[
 E C_i=p,\quad \operatorname{Var}(C_i)=p(1-p),\quad
 \operatorname{Cov}(C_i,C_j)=-\frac{p(1-p)}{d-1}\quad(i\ne j).
\]

因此

\[
 R_C=\frac d{d-1}I-\frac1{d-1}J.
\]

常数向量方向的特征值为0，正交子空间的特征值为d/(d-1)，所以必须使用lambda_max=d/(d-1)，不是最小特征值0。有限iid严格下降方向存在当且仅当K>d；K<d为严格局部极小，K=d是未由Hessian解决的临界点。这个门槛与m无关，只在非退化边际内成立。

总依赖恰为

\[
 D_{KL}(P\Vert\operatorname{Ber}(p)^{\otimes d})
 =d h(p)-\log {d\choose m}
 =-\log\Pr\{\operatorname{Bin}(d,p)=m\}.
\]

若p沿d趋于某个(0,1)内常数，Stirling公式给总依赖为(1/2)log[2*pi*d*p(1-p)]+o(1)。由于其余d-1位完全决定最后一位，I(C_i;C_-i)=h(p)。若m固定而d增加，p趋于0，此单概念信息也趋于0；不能省略该尺度条件。

**模型边界。** 有限mu和协方差的Gaussian-logistic-Bernoulli模型对每个二值状态均赋正概率，不能精确生成只支持一个count的分布。这是损失几何的misspecified反例，不是正确指定模型失败的证明。接下来的软总量抑制族解决这一缺口，但不具有精确总数约束。

## 3. 正确指定的正/负等相关Gaussian-logistic族

固定sigma>0、0<c<1，并令rho_d=c/(d-1)。定义

\[
 R_{d,+}=(1-\rho_d)I+\rho_d J,\qquad
 R_{d,-}=(1+\rho_d)I-\rho_d J.
\]

两者对角均为1。正族特征值为1+c（一次）及1-c/(d-1)（d-1次）；负族为1-c（一次）及1+c/(d-1)（d-1次），所以均严格正定。

真分布直接由

\[
 U\sim N(0,R_{d,\pm}),\quad
 C_i\mid U\stackrel{ind}{\sim}\operatorname{Ber}(\sigma_{\rm log}(\sigma U_i))
\]

生成。它就在被拟合的Gaussian-logistic族内，真参数为mu=0、Sigma=sigma^2 R_(d,sign)。这里**mu必须是零向量**；若固定非零mu，一般不能保持下面精确的正负等幅性质。所有边际恒为1/2，与sigma、c、d和相关符号无关。

令Y_i=2C_i-1，f(z)=tanh(sigma*z/2)。条件均值E[Y_i|U_i]=f(U_i)。定义

\[
 r_d=E[f(Z_1)f(Z_2)],\quad
 (Z_1,Z_2)\sim N\!\left(0,\begin{pmatrix}1&\rho_d\\\rho_d&1\end{pmatrix}\right).
\]

由于f为奇函数，把第二个Gaussian坐标反号即得负族的观测二值相关为-r_d；正族为+r_d。这是精确等式。把f展开于标准Gaussian的正交归一Hermite基，记a_l=E[f(Z)H_l(Z)/sqrt(l!)]，则

\[
 r_d=\sum_{l\text{ odd}}a_l^2\rho_d^l,
 \qquad a_1=E[Zf(Z)]=E[f'(Z)]>0.
\]

令beta=c*a1^2。Parseval给sum a_l^2=E f(Z)^2<1，所以

\[
 r_d=\frac{\beta}{d-1}+O((d-1)^{-3}),\qquad
 0<r_d\le\frac c{d-1}E f(Z)^2<\frac c{d-1}.
\]

特别地0<beta<c<1。更强的a1^2<=2/pi界可由Z*tanh(sigma Z/2)<=|Z|得到，但下文只需beta<1。

### 3.1 采样门槛的符号/结构不对称

正族的观测相关最大特征值为1+(d-1)r_d；负族为1+r_d（d-1重），其总量方向的小特征值为1-(d-1)r_d。代入共同局部公式：

\[
 K_{\rm crit,+}=1+\frac1{(d-1)r_d}
       =1+\frac1\beta+O(d^{-2}),
\]
\[
 K_{\rm crit,-}=1+\frac1{r_d}
       =1+\frac{d-1}{\beta}+O(d^{-1}).
\]

整数K严格大于相应实数临界值才有负曲率；小于时独立点是严格局部极小。正/负族有相同边际、完全相同的每对互信息，却有常数阶与线性d阶的门槛。这里没有使用lambda_min冒充lambda_max。

对固定K和增长d，负族最终落在严格局部极小区；正族只要K>1+1/beta则最终为鞍点。Z/-Z配对在两个族中都把独立点变为鞍点，但其最弱负曲率仍可随d消失，不能据此声称训练速度不随d变差。

### 3.2 d=128、c=.9、K=100不依赖数值积分的判断

由r_d<c/(d-1)，负族满足

\[
 r_{128}<.9/127<1/99,
\]

因此K=100的iid独立点严格局部极小，**对任何固定有限sigma>0均成立**。不需要先计算sigma=4的积分，也不依赖Kcrit大约300这一未经本文件数值认证的估计。

若sigma=4，则a1=E[Z*tanh(2Z)]满足解析下界

\[
 a_1\ge E[|Z|1_{|Z|\ge1/2}]\tanh(1)
       =2\varphi(1/2)\tanh(1).
\]

由此beta=.9*a1^2>1/99。又(d-1)r_d>=beta，所以正族在同一d=128、K=100下有负曲率。此处仅作标准函数常数不等式，不做拟合或数值积分。

**解释。** 总量抑制被分散到d-1个微弱的正相关特征模态偏离上；公共因子则把相关性集中到一个正特征模态。有限MC的局部代价按这些正偏离筛选可学方向。该结构区别比只说“相关小所以难学”更具体。

## 4. 两两信息小，不等于总信息消失

每对的相关量是Theta(1/d)，**不是O(1/d^2)**。每对的互信息才是

\[
 I(C_i;C_j)=\tfrac12[(1+r_d)\log(1+r_d)+(1-r_d)\log(1-r_d)]
           =\tfrac12r_d^2+O(r_d^4)=\Theta(d^{-2}).
\]

由于该式关于r为偶函数，正负族每对互信息精确相同。

完整观察分布交换对称。记N=sum C_i、Q0=Ber(1/2)^d。给定N=n，P和Q0都在同一count切片上均匀，故KL链式分解精确给出

\[
 D_{KL}(P_{d,\pm}\Vert Q_0)
 =D_{KL}(\mathcal L_{P_{d,\pm}}(N)\Vert\operatorname{Bin}(d,1/2)).
\]

这避免枚举2^d状态，但实际估计count分布仍有尾部与有限样本偏差问题；本文件不声称任意经验直方图即可精确算KL。交换对称与KL链式规则是标准工具。

总自旋S_d=sum Y_i的方差精确为

\[
 \operatorname{Var}(S_d)=d[1\pm(d-1)r_d]=d(1\pm\beta)+O(d^{-1}).
\]

计数N的方差为其四分之一。因此负族把计数方差降低一个非消失的固定比例beta，但方差仍是Theta(d)，并非固定计数。

## 5. 可以证明CLT，而非只凭方差猜测

下面给出适用于固定sigma、c的证明。所用是iid多元CLT、Taylor展开及有界条件独立Bernoulli的条件CLT。

### 5.1 负族

取iid Z_1,...,Z_d~N(0,1)，barZ=d^(-1)sum Z_i，令

\[
 a_d=\sqrt{1+c/(d-1)},\quad s=\sqrt{1-c},\quad b_d=a_d-s,
 \qquad U_i=a_d Z_i-b_d\bar Z.
\]

直接计算可核对Cov(U)=R_(d,-)，不是近似表示。由于a_d-1=O(1/d)、barZ=O_p(d^(-1/2))，f的一、二阶导数有界，Taylor展开给

\[
 \frac1{\sqrt d}\sum_i f(U_i)
 =\frac1{\sqrt d}\sum_i f(Z_i)
  -(1-\sqrt{1-c})a_1\frac1{\sqrt d}\sum_i Z_i+o_p(1).
\]

余项的理由：各位移平方之和为O_p(1)，除sqrt(d)后趋零；a_d-1乘以sum|Z_i|/sqrt(d)也是o_p(1)；平均f'(Z_i)依大数定律趋于a1。

再把Y_i分解为f(U_i)+epsilon_i。给定U，epsilon_i独立、中心且绝对值<=2，条件方差平均趋于1-E f(Z)^2。条件特征函数展开的三阶余项总计O(d^(-1/2))，故其归一化和收敛到与Gaussian输入极限独立的N(0,1-E f^2)。同时iid多元CLT给(sum f(Z_i),sum Z_i)/sqrt(d)的极限协方差为

\[
 \begin{pmatrix}E f^2&a_1\\a_1&1\end{pmatrix}.
\]

令b=1-sqrt(1-c)，利用2b-b^2=c，合并得到

\[
 S_d/\sqrt d\Rightarrow N(0,1-ca_1^2)=N(0,1-\beta).
\]

### 5.2 正族

用独立Z0和iid Zi表示U_i=sqrt(1-rho_d)Zi+sqrt(rho_d)Z0。相同Taylor论证给

\[
 \frac1{\sqrt d}\sum_i f(U_i)
 =\frac1{\sqrt d}\sum_i f(Z_i)+a_1\sqrt c\,Z_0+o_p(1).
\]

加上同样的Bernoulli条件噪声，得S_d/sqrt(d)趋于N(0,1+beta)。两族的有限维极限都可以局部趋于独立，而其总量的极限方差不同于独立模型。

## 6. 全局依赖Theta(1)：用DPI夹住，不误用CLT推出熵收敛

**下界。** 固定事件A_d={|S_d|<=sqrt(d)}。在Q0下其概率趋于b0=2Phi(1)-1；在P_(d,sign)下趋于b_sign=2Phi(1/sqrt(1+sign*beta))-1。由于beta>0，两极限不同，且均严格位于(0,1)。由数据处理与有限字母KL连续性，

\[
 \liminf_d D_{KL}(P_{d,\pm}\Vert Q_0)
 \ge D_{KL}(\operatorname{Ber}(b_\pm)\Vert\operatorname{Ber}(b_0))>0.
\]

**上界。** 将同一独立Bernoulli发射核作用于N(0,R_(d,sign))和N(0,I)，后者恰产生Q0。数据处理给

\[
 D_{KL}(P_{d,\pm}\Vert Q_0)
 \le -\tfrac12\log\det R_{d,\pm}.
\]

负族右侧趋于[-log(1-c)-c]/2，正族趋于[c-log(1+c)]/2，均为有限常数。所以固定sigma>0和c∈(0,1)时，总KL是Theta(1)。

这里**没有证明**总KL收敛到Gaussian方差KL `(v-1-log v)/2`；弱收敛不足以推出相对熵收敛，若要该精确极限需另有局部CLT及尾部/熵控制。当前上下界已足够证明非消失。DPI、Gaussian KL和有限事件粗化均不作为新理论工具。

也不能把Theta(1)描述为每个概念都有常数收益：总KL除以d会趋于0。正负族的完整KL不保证相等，本文件仅保证其同阶非消失以及每对互信息精确匹配。

## 7. 半数反馈具有非消失信息，可被固定伙伴评价遗漏

令A、B是预先固定的不交分组，|A|/d趋于tau∈(0,1)，|B|=d-|A|。定义T_A=sum_(i∈A)Y_i/sqrt(d)，T_B同理。上节联合CLT的同一证明给

\[
 (T_A,T_B)\Rightarrow N\!\left(0,
 \begin{pmatrix}
 \tau\pm\beta\tau^2&\pm\beta\tau(1-\tau)\\
 \pm\beta\tau(1-\tau)&(1-\tau)\pm\beta(1-\tau)^2
 \end{pmatrix}\right).
\]

负族可通过同一个全局W=sum Zi/sqrt(d)的修正获得所有交叉项；条件Bernoulli噪声在不交分组之间渐近独立。正族则由共享Z0产生交叉项。

极限相关为

\[
 r_{\rm block,\pm}=\frac{\pm\beta\sqrt{\tau(1-\tau)}}
 {\sqrt{(1\pm\beta\tau)(1\pm\beta(1-\tau))}}.
\]

tau=1/2时，分别为+beta/(2+beta)和-beta/(2-beta)，均非零。注意这个block相关不同于每对概念的r_d。

固定粗化事件U_A=1{T_A>0}、U_B=1{T_B>0}。连续非退化Gaussian极限在坐标轴上无质量，故象限概率收敛，阈值处有限d的平局不影响极限。标准Gaussian象限式给两个符号位的极限相关

\[
 \kappa_\pm=\frac2\pi\arcsin(r_{\rm block,\pm}).
\]

因此

\[
 \liminf_d I(C_A;C_B)
 \ge \lim_d I(U_A;U_B)
 =\log2-h((1+\kappa_\pm)/2)>0.
\]

这里信息下界来自仅4状态的粗化事件，未声称连续极限的Gaussian MI等于离散count MI。交换对称还给出有限d精确关系I(C_A;C_B)=I(N_A;N_B)：给定两边count，两块内部配置分别均匀且相互独立；组合数项在互信息中抵消。

**反馈用途。** 若A的概念被精确观察，用真条件分布预测B的固定事件U_B，log loss的最优改进为I(C_A;U_B)，其liminf至少为上述正数。独立模型不会传播A的信息。因而本族存在非消失的block-event反馈价值；不需要把类标签或额外任务头引入定义。

但这个结论要求查询固定正比例概念。对固定少量查询、固定少量伙伴，有限维分布趋于独立，每对信息是Theta(d^-2)。不能把半数反馈的常数信息外推成单查询的常数收益。旧单查询/两伙伴指标的低收益与这里的全局计数结构可以并存；这是评价对象不同，不是旧结果算错。

## 8. 自由方差也不能把负相关族压成常数规模的潜变量

这一节约束的是**固定标准sigmoid发射所用的Gaussian logit坐标**。若允许任意重缩放latent再反向放大发射增益，原始latent方差可以任意改单位；下列界必须写在最终logit的协方差Sigma上，不能称所有Gaussian表示下无坐标条件的不变量。

设某个Gaussian-logistic-Bernoulli表示具有全部观察边际1/2、所有二值spin pair correlation均为-r<0，但不要求潜在Sigma交换对称，也不要求方差相等。

### 8.1 任意此类表示都满足mu=0与线性trace下界

对固定有限方差v，E sigmoid(mu+sqrt(v)Z)关于mu严格递增，且在mu=0恰为1/2。因此每个观察边际1/2都强迫对应mu_i=0；自由均值不能逃避这个条件。

固定两维方差vi、vj，记潜在协方差为s。Price恒等式及0<sigmoid'<=1/4给

\[
 \frac{d}{ds}E[\operatorname{sigmoid}(\eta_i)\operatorname{sigmoid}(\eta_j)]
 =E[\operatorname{sigmoid}'(\eta_i)\operatorname{sigmoid}'(\eta_j)]\in(0,1/16].
\]

从s=0积分，并连续延伸到PSD边界，得到观察协方差与s同号、且绝对值不超过|s|/16。由于Cov(C_i,C_j)=-r/4，必有

\[
 \Sigma_{ij}\le-4r\quad(i\ne j).
\]

PSD又要求0<=1^T Sigma 1=tr Sigma+sum_(i!=j)Sigma_ij，故

\[
 \boxed{\operatorname{tr}\Sigma\ge4r\,d(d-1).}
\]

当r=r_d=beta/(d-1)+O(d^-3)时，右侧为4*beta*d+O(d^-1)=Omega(d)。这个界覆盖所有满足给定观察pair moments的Gaussian-logistic表示，当然也覆盖同一完整观察分布的所有此类表示；不需要潜在协方差等相关。

### 8.2 实际还有rank>=d-1的约束

所有潜在非对角元素均严格为负。若rank(Sigma)<=d-2，则其核空间维数至少2，可在核中取非零w且sum w_i=0。写w=w_+-w_-，两非负部分均非零且支持不交。由Sigma*w=0得到

\[
 w_+^T\Sigma w_+=w_+^T\Sigma w_-.
\]

左侧由PSD非负，右侧因跨支持的Sigma_ij全为负而严格负，矛盾。因此

\[
 \boxed{\operatorname{rank}\Sigma\ge d-1.}
\]

该线性代数论证不作为新的一般rank定理；它给当前表示合同一个额外的可检查后果：负等相关不能通过任意非交换的低秩Gaussian因子来规避。

### 8.3 匹配正相关可以由rank-one、常数trace的另一真分布实现

令所有eta_i=sqrt(v)Z，共享一个标准Gaussian。其观察spin相关为

\[
 r(v)=E\tanh^2(\sqrt v Z/2).
\]

它从0到1连续严格递增，所以任意0<r<1对应唯一v；小v时r(v)=v/4+O(v^2)，故v=4r+O(r^2)。取r=r_d，则

\[
 \operatorname{rank}\Sigma=1,\qquad
 \operatorname{tr}\Sigma=dv=4\beta+O(d^{-1})=O(1).
\]

**对象区别不能略去。** 这个rank-one正族与第3节固定sigma正等相关族有同一观察边际及pair correlation，但一般不是同一个完整观察联合分布。这里构造的是一个moments匹配的新正对照，不证明原P_(d,+)可精确压缩成rank-one表示。允许半正定Gaussian即可精确实现rank-one；若某源码强制严格正对角，这又是其闭包中的对象，需单独处理实现约束。

这个新正对照也有Theta(1)的总依赖。其count CLT仍为S_d/sqrt(d)趋于N(0,1+beta)：给定Z，其归一化条件均值趋于sqrt(beta)Z，条件Bernoulli噪声趋于独立N(0,1)。于是第6节固定事件给非消失KL下界。

上界不使用奇异Gaussian相对独立Gaussian的KL，而用混合凸性：

\[
 D_{KL}(P_{d,+}^{\rm rank1}\Vert Q_0)
 \le d\,E D_{KL}(\operatorname{Ber}(\operatorname{sigmoid}(\sqrt v Z))\Vert\operatorname{Ber}(1/2))
 \le dv/8=O(1).
\]

最后一个不等式由k(a)=KL(Ber(sigmoid(a))||Ber(1/2))满足k'(a)=a*sigmoid'(a)、k(0)=0及sigmoid'<=1/4得到，且k为偶函数。这避免了把奇异协方差的latent KL误用为有限上界。

### 8.4 与有限MC惩罚的正确连接

对协方差路径tS，额外iid一阶项为tr(S Gamma)/(2K)。负族Gamma的最小特征值为[1-(d-1)r]/4，趋于(1-beta)/4>0。因此若S本身是任何实现该负pair矩的logit协方差，前述trace下界意味着这条路径的MC额外系数至少是Omega(d/K)。rank-one正对照的相应量为O(1/K)。

这解释了全局结构分散到许多潜在方向时，固定MC预算面临的不同代价。但不能从这个局部系数直接推出“任意有限方差处的期望目标都更差”，也不能据此认证独立模型优于某个真参数点的finite-K全局目标。后者若要作witness，须在独立冻结的数值或进一步解析工作中比较真实人口目标。

第3节所有下降方向的精确门槛仍由lambda_max控制；本节trace/rank下界是补充的表示合同，不应替代该判据或改写为lambda_min学习阈值。

### 8.5 trace/rank下界其实不要求平衡边际

设任意非退化观察边际p_i∈(0,1)，D_i=p_i(1-p_i)。只在本小节不再假定mu=0。沿Price协方差路径改变Sigma_ij，保持两个一维Gaussian边际的mu_i、v_i不变，故观察p_i也不变。

记q_i=g(eta_i)、g_i'=q_i(1-q_i)。逐点有(g_i')^2<=g_i'/4，而

\[
 E g_i'=p_i-Eq_i^2\le p_i-p_i^2=D_i.
\]

所以沿整条合法协方差路径，由Cauchy-Schwarz得到

\[
 0<E[g_i'g_j']
 \le\sqrt{E(g_i')^2 E(g_j')^2}
 \le\frac14\sqrt{D_iD_j}.
\]

协方差为0时两个观察独立；从0积分至Sigma_ij，除以sqrt(D_iD_j)，得到与符号一致的界

\[
 \boxed{|\operatorname{Corr}(C_i,C_j)|\le |\Sigma_{ij}|/4.}
\]

因此若所有观察pair correlation均<=-r<0，依然有Sigma_ij<=-4r，并且

\[
 \operatorname{tr}\Sigma\ge4r d(d-1),\qquad
 \operatorname{rank}\Sigma\ge d-1.
\]

若各对负相关强度不同，还可写tr Sigma>=4 sum_(i!=j)|Corr(C_i,C_j)|，前提仍是所有相关为负；rank结论使用严格负号。这个一般化不证明任意不平衡边际都能由前述rank-one正族精确匹配。比较的正对照构造及第3—7节的对称分布结论仍限于平衡边际。

## 9. 信息开销：并非通过自由方差或非识别表示即可消除

本节的P_d指第3节正确指定的负族，T_d=TC(P_d)=Theta(1)，r=r_d>0。讨论完整同一P_d的任意Gaussian-logit表示，不仅讨论原等相关参数。

### 9.1 最小方差子集给出的有限d界

将任一表示的latent logit方差排序为v_(1)<=...<=v_(d)，取m=ceil(d/2)。前m个变量的观察相关仍全为-r，对其principal covariance应用第8节trace界：

\[
 \sum_{j=1}^m v_{(j)}\ge4r m(m-1),
 \qquad v_{(m)}\ge4r(m-1).
\]

因此至少d-m+1维的方差不小于4r(m-1)，该下限趋于2*beta>0。定义

\[
 g(v)=\log2-E h(\operatorname{sigmoid}(\sqrt v Z)).
\]

g(0)=0，v>0时g(v)>0且随v严格递增：binary entropy关于logit绝对值严格递减，|sqrt(v)Z|逐点增大；连续性由有界收敛成立。

所有边际1/2强迫mu=0，且给定eta后各C_j独立，所以准确恒等式为

\[
 I(C;\eta)=H(C)-E H(C\mid\eta)
          =\sum_{j=1}^d g(v_j)-T_d.
\]

结合第6节的有限d上界

\[
 T_d\le U_d(c):=-\tfrac12\left[\log(1-c)+(d-1)\log(1+c/(d-1))\right],
\]

得到对所有正确表示统一成立的界

\[
 I(C;\eta)\ge
 (d-m+1)g(4r_d(m-1))-U_d(c)=\Omega(d).
\]

U_d(c)有界且趋于[-log(1-c)-c]/2；例如所有d>=2时U_d(c)<=-(1/2)log(1-c)。这是有限常数明确的下界。它不自动保证在任一给定d上减去log(K)后仍为正，具体有限d比较须代入已认证的常数。

### 9.2 更强的独立推导：其实覆盖所有条件独立发射表示

令W是任意变量，只要求给定W后C_1,...,C_d条件独立，不要求Gaussian、线性logit、固定latent维数或相等方差。记

\[
 a_i(W)=E[2C_i-1\mid W],\qquad G=E[aa^T].
\]

G为PSD；条件独立及平衡边际给G_ij=E[(2C_i-1)(2C_j-1)]=-r_d（i!=j）。因此

\[
 0\le1^TG1=\sum_i E a_i^2-r_d d(d-1),
 \qquad\sum_i E a_i^2\ge r_d d(d-1).
\]

Bernoulli的Pinsker不等式（自然对数）给

\[
 I(C_i;W)=E D_{KL}(\operatorname{Ber}((1+a_i)/2)\Vert\operatorname{Ber}(1/2))
 \ge\tfrac12 E a_i^2.
\]

又由条件独立的熵恒等式I(C;W)=sum_i I(C_i;W)-T_d，得到更直接的有限d界

\[
 \boxed{I(C;W)\ge\tfrac12r_d d(d-1)-T_d
             \ge\tfrac12r_d d(d-1)-U_d(c).}
\]

右侧为beta*d/2-O(1)。这个证明甚至不依赖第9.1节的方差坐标，可直接排除改用非Gaussian latent来消除信息开销的做法，**但条件独立发射约束不可去掉**。

对W取所有条件独立表示的infimum，就是多元Wyner common information的标准定义。因此本节是该已知量的一个特定binary源下界；不把它作为新信息概念。负族的该值为Theta(d)：下界如上，而W=C给平凡上界H(C)<=d*log2。对于第8.3节新的pair-matched rank-one正族，显式表示满足

\[
 I(C;\eta)=d g(v)-TC(P_{d,+}^{\rm rank1})\le d g(v)\le dv/8=O(1).
\]

其非消失block MI也给common information正下界，所以该正对照的common information为Theta(1)。这段标准sigmoid/rank-one构造仍是**选定的新观察分布**，不证明原固定sigma正族存在同一完整观察分布的Gaussian-logit低开销表示。

这给出一个较强的结构分离：观察TC在两个族中都是Theta(1)，每对MI精确匹配，但生成所有条件独立输出所需的最小共同latent信息为Theta(d)与Theta(1)。相同order不等于观察TC的数值精确相等。

还可对第3节原固定sigma正族作一个独立的unrestricted common-information观察：其U_i=sqrt(1-rho)Z_i+sqrt(rho)Z0。把独立Z_i积分掉后，C_i给定Z0仍条件独立，发射均值是一个Gaussian平滑后的sigmoid函数。因此

\[
 I(C;Z_0)\le I(U;Z_0)
 =\tfrac12\log\left(1+\frac{d\rho_d}{1-\rho_d}\right)=O(1).
\]

结合block MI下界，原P_(d,+)的unrestricted common information也为Theta(1)。但平滑后的发射函数一般不是标准sigmoid作用在线性Gaussian logit上，故这不是第8节标准link表示的rank-one观察等价证明。它说明正族的私有Gaussian噪声可以在条件发射中解析边际化；负族的任何条件独立latent信息下界排除了同样的低信息表示。这里区分的是表示/发射合同，不能把原完整eta的posterior与边际化后Z0的posterior当成相同推断目标。

### 9.3 子集反馈也可有线性latent信息，但与可见反馈效用不同

若只观察一个固定大小m的子集A，同一Gram/Pinsker证明给

\[
 I(C_A;W)\ge\tfrac12r_d m(m-1)-TC(C_A)
           \ge\tfrac12r_d m(m-1)-T_d.
\]

最后一步用TC随加入变量非减。m/d趋于tau∈(0,1)时，这仍为(beta*tau^2/2)d-O(1)。固定有限m则没有这样的线性结论。注意，I(C_A;W)的线性规模是latent posterior所需的信息；第7节观察块之间的反馈信息仅证明Theta(1)量级下界。二者不可混作同一个信息量。

## 10. 固定prior候选选择的前向后验KL障碍

令pi是任一正确表示W的prior，p(dw|c)是真posterior。对每个c，先产生K个候选W_1,...,W_K，每个候选在给定c时的边缘分布都是pi；再用依赖c和候选的任意随机选择规则输出其中一个。标准iid prior SIR是特例。

这里q_K(dw|c)是**对随机candidate bank及选择随机性均积分后的输出点分布**。不是固定一次抽样后只有K个原子的经验测度；后者对连续真posterior的forward KL通常直接无穷，不能作为有意义的样本复杂度结论。

对任意可测集合A，输出落在A必要求至少一个候选落在A，故

\[
 q_K(A\mid c)\le\sum_{k=1}^K\Pr(W_k\in A\mid c)=K\pi(A).
\]

因此q_K相对pi的Radon-Nikodym导数至多K。这个测度形式也覆盖奇异Gaussian及rank-one表示，不要求存在全空间Lebesgue密度。继而

\[
 D_{KL}(p(W\mid c)\Vert q_K(W\mid c))
 \ge D_{KL}(p(W\mid c)\Vert\pi)-\log K.
\]

若q_K在真posterior有质量之处取零，左侧为无穷，不等式仍成立。对正确P_d中的c取平均得

\[
 \boxed{E_C D_{KL}(p(W\mid C)\Vert q_K(W\mid C))
       \ge I(C;W)-\log K
       \ge\tfrac12r_d d(d-1)-U_d(c)-\log K.}
\]

所以固定sigma、c的负族中，任何多项式K仍留下Omega(d)平均前向后验KL；若要该平均KL为o(d)，至少需要log(K)=Omega(d)。对固定比例的子集反馈，同样用第9.3节的I(C_A;W)得到对应结论。正rank-one对照不受这个线性下界约束，但这本身不证明其有限K误差小或给出所有目标的充分采样预算。

**独立性并非必要。** 上面的union bound只用每个候选给定c时的边缘为pi，故Z/-Z的antithetic候选同样受到限制。这与配对消除独立点局部极小不矛盾：局部训练曲率与完整latent posterior的近似是不同问题，配对不是所有有限MC误差的全局修复。

**严格限定。** 不覆盖依赖观测c的良好proposal、transport、MCMC产生新点、直接count推断或其他非候选选择算法；不能说所有推断都需要指数预算。候选的无条件边缘为pi还不够，必须是固定c后的候选边缘为pi，否则真posterior样本对C平均也可边缘等于prior而K=1已经准确。

本界是 `true posterior || selected-point law` 方向的KL，**绝不是**F_K期望训练gap的下界。IWAE/增广分布恒等式涉及不同KL方向和空间，不能自动换用。大latent forward KL也不推出大观察事件风险：数据处理只给相关观测误差的上界方向，粗化后仍可能准确。训练gap、候选选择后验误差与block反馈预测损失需要分别证明或核验。

## 11. 文献最强重合与当前可守增量

查阅日期2026-09-29；没有下载本地全文附件或执行论文复现。

- [Schneidman等，Nature 2006，Weak pairwise correlations imply strongly correlated network states](https://www.nature.com/articles/nature04701)：已读原始摘要及作者稿可见内容。弱两两相关与强群体效应并存、用联合分布和信息描述集体状态，是很早已有的主题，不能作为本轮一般性发现。
- [Macke等，Neural Computation 2009，Generating Spike Trains with Specified Correlation Coefficients](https://www.mackelab.org/pubs/Macke_Berens_2009.pdf)：已读潜在Gaussian构造、相关可行性以及群体count部分。已有Gaussian产生相关二值向量、正/负相关改变总数离散度及弱相关影响群体count的研究。其dichotomized Gaussian发射不同于本轮有限sigma的logistic-Bernoulli；不能仅以换发射链接宣称原创。
- [Galicza、Pete，Sparse reconstruction in spin systems II](https://link.springer.com/article/10.1007/s00440-025-01395-4)：已读摘要、引言和相关定义，未审核全文全部定理。已有从部分自旋获取全局事件信息、Curie-Weiss结构与entropy/clue的系统理论。其重点包括o(d)查询，而本文证明的是固定比例查询；不能称首次研究全局约束的反馈/重建信息。
- [Daudel等，JMLR 2023，Alpha-divergence Variational Inference Meets Importance Weighted Auto-Encoders](https://www.jmlr.org/papers/volume24/22-1160/22-1160.pdf)：已读摘要/背景及高维结果的概述，未逐条审核83页证明。多样本目标的维数灾难及样本预算增长已有理论。本轮区别应是**同维、同边际、同绝对pair information下，正/负结构导致常数/线性d的局部学习门槛**，不是一般“高维需要更多MC”。
- 前轮FIVO/JVI、variance-component score、谱collapse及antithetic近邻继续适用，见前轮文献表 (historical reference not distributed)及补充近邻 (historical reference not distributed)。Hermite/Mehler恒等式、CLT、Gaussian象限公式、KL链式/DPI和等相关矩阵特征值均是标准工具。

### 信息开销加强后的直接近邻

- [Sula、Gastpar，Common Information Components Analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC7912312/)：已读Wyner定义及第7节多元扩展。对所有使各变量条件独立的W最小化I(C;W)，正是已知的多元Wyner common information；本轮不能将该量重新命名成新概念。本文没有完整审核其余算法与全部证明。
- [Liu、Courtade、Cuff、Verdu，ISIT 2016，Brascamp-Lieb Inequality and Its Reverse: An Information Theoretic View](https://people.eecs.berkeley.edu/~courtade/pdfs/Liu_ISIT2016a.pdf)：已读摘要及式(29)。该文给多元Gaussian源的common information为 `(1/2) inf_{diagonal Lambda <= Sigma} log(det Sigma/det Lambda)`。对等相关矩阵，置换对称及log-det凹性使最优Lambda可取lambda_min(Sigma)*I。将第3节R_+与R_-代入，得到正族 `(1/2)log[(1+c)/(1-c/(d-1))]=O(1)`，负族 `(d-1)/2 log[(1+c/(d-1))/(1-c)]=Theta(d)`；两者连续Gaussian TC均O(1)。这段代入是本轮对其标准公式的推论，说明**负相关结构的高common-information开销至少在连续Gaussian中已有直接理论基础**，不是本项目首次发现的一般现象。
- [Chatterjee、Diaconis，The sample size required in importance sampling](https://arxiv.org/abs/1511.01437)，Annals of Applied Probability 2018：已读作者摘要及定理概述。用target/proposal KL解释importance sampling所需样本的指数尺度已有；具体必要/充分结论带其正则与尾部条件。本轮第10节是特定候选选择输出的forward-KL界，不能取代或宣称首次给出一般exp(KL)采样复杂度。
- [Sanz-Alonso，Importance Sampling and Necessary Sample Size: An Information Theory Approach](https://epubs.siam.org/doi/10.1137/16M1093549)：已读出版社摘要。已有基于f-divergence的信息论非渐近必要样本条件，涵盖KL等；正文未完整审核，是仍需正面对照的近邻。
- [Cremer、Morris、Duvenaud，Reinterpreting Importance-Weighted Autoencoders](https://arxiv.org/abs/1704.02916)：已读原始摘要，并核对[TensorFlow Probability官方ImportanceResample定义](https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/distributions/ImportanceResample)。SIR的隐式/平均输出分布与IWAE变分解释已有；不能把平均selected-point law当作新推断对象。本文件刻意保留forward-KL与训练gap的方向区别。

信息开销加强后，最强的已有工作反驳更明确：common information与importance sampling复杂度都是成熟主题，连续Gaussian的正负结构开销分离也能由既有公式直接推出。本轮若有增量，应定位在**正确指定的binary-logistic观察族中，匹配pair信息却有非消失全局结构，对任何条件独立表示的有限d信息下界，以及它与特定有限MC训练局部几何和prior候选后验误差的联合刻画**。未检到完整相同链条不等于原创性已认证。

目前更有辨识度的候选是：**有限MC目标并非只忽略“整体信息太弱”的依赖；它对依赖的谱组织方式有结构性偏好。在正确指定模型、匹配边际和pair信息的两族中，非消失的集体信息可以伴随线性维数的独立局部极小门槛，而公共因子族仅需常数预算。**

这比先前弱相关二元玩具更像可独立组织的理论论文核心，因为它回答了“被压掉的是否只是可忽略信息”的反驳，并把局部优化、可实现真分布与不同评价尺度接在一起。但它仍是候选：本文件没有完成穷尽新颖性检索，也没有证明实际优化器学习到对应分布、配对训练恢复block信息、或自然概念系统常有该负等相关结构。

第9—10节进一步排除了“自由方差或另一等价latent表示总能把信息开销降为常数”的反驳；其代价是必须承认配对仅消除某个局部门槛，仍受prior候选选择的全局posterior障碍约束。这个区分使机制更完整，但也禁止把antithetic包装成完整解决方案。

最强拒稿意见仍可能是“把已有谱判据代入经典等相关矩阵，再结合已知弱相关/集体信息现象，是直接推论”。若写论文，必须以具体结构敏感的采样复杂度分离为主张，而不是用一般信息不等式或熟知的计数约束充当原创。有限K严格局部极小的邻域可随d变化、配对负曲率可趋于零、全局最优与恢复速度均未被认证。全局依赖、两两依赖、单查询效用和成块反馈效用必须始终分别报告。
