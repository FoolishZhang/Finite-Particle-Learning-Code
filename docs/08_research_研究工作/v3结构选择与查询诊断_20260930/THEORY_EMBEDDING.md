# 固定高阶块的维数嵌入：表示下界与可学习性

本文件是独立解析追加包T2。它不修改已冻结的THEORY_DESIGN、T包方案、源码或结果，不产生任何新观察、拟合或随机数。所有KL采用自然对数，joint KL是完整向量的总量，不除以维数。正文给出的显式常数来自符号推导，没有用有限box数值替代无界模型证明。

本文只研究普通、单一结构、无隐变量混合的laminar recursive-cardinality（RC）分布，以及一个已知三对差分模板的稀疏高阶教师族。RC、Fourier矩恢复、Hoeffding和KL链式均为经典工具；这里不声明新的一般信息论或新的稀疏Fourier学习算法。

## 1. 模型、量词与固定六变量见证

在二元变量C上，普通fullsupport laminar RC指

\[
 Q(c)=Z^{-1}\exp\left\{
 \sum_{i=1}^{D}a_i c_i+
 \sum_{S\in\mathcal L}g_S\left(\sum_{i\in S}c_i\right)
 \right\},
 \tag{1}
\]

其中所有参数有限，g_S是任意实值函数，集合族L中任意两个节点嵌套或不交。节点、结构及参数都可自由选择；本文件不对势设置有限box。单点势可并入unary，常数势可并入Z。允许加入全根节点，其势可以恒为零。

先固定六个坐标，配对为(1,2)、(3,4)、(5,6)，定义

\[
 h(c)=(c_1-c_2)(c_3-c_4)(c_5-c_6),\qquad
 P_\epsilon(c)=2^{-6}[1+\epsilon h(c)],
 \quad 0<\epsilon<1.
 \tag{2}
\]

主见证固定epsilon=9/10。h取值为−1、0、1，只有每对恰有一个1时才非零。在均匀分布U_6下，这个事件的概率为1/8；其八种取向中h的正负各占一半。因此(2)归一化且fullsupport。

对任意D>=6，定义嵌入

\[
 P_{\epsilon,D}=P_\epsilon\otimes U_{D-6},
 \tag{3}
\]

其中前六维是固定活跃块，其余位独立公平。本节至第6节的查询工作负载也是预先固定的前六维，不能随观察到的教师结构事后选择。

## 2. 低阶与全count统计仍然看不见该依赖

令Y_i=2C_i−1。则

\[
 h=2^{-3}(Y_1-Y_2)(Y_3-Y_4)(Y_5-Y_6).
 \tag{4}
\]

这是纯三阶Walsh多项式。因此P_epsilon的所有少于三维的边际与U_6完全相同：单变量和所有pair都均匀独立。嵌入(3)保留此性质。

h非零时活跃块count恒为3；在该count层内，八种取向的h和为零。因此P_epsilon的活跃块count分布与Binomial(6,1/2)相同，嵌入后的完整count为Binomial(D,1/2)。

更强地，对任何次数不超过2的坐标乘积phi及任何全count n，

\[
 E_{U_D}[h(C)\phi(C)\mathbf 1_{\{N=n\}}]=0.
 \tag{5}
\]

理由是在h非零的取向立方体上，N_active恒为3，phi最多接触两对，至少一个未接触配对的取向求和使h抵消；再对非活跃变量求和即可。故pair与全count的联合统计也与U_D相同。这里说的是相对于均匀count条件分布的残差为零，绝不是说E[Y_iY_j|N]本身为零。

这说明“所有单变量、pair及完整count统计匹配”不构成完整分布或条件查询正确性的证书。它不排除三阶或更高阶诊断识别该教师。

## 3. 无界laminar RC的定量六维障碍

记

\[
 A=\tfrac12\log(1-\epsilon^2)<0,\qquad
 B=\operatorname{atanh}\epsilon>0.
\]

由于h属于{−1,0,1}，

\[
 \log P_\epsilon=-6\log2+A h^2+B h,\qquad
 h^2=2^{-3}\prod_{a=1}^{3}(1-Y_{2a-1}Y_{2a}).
 \tag{6}
\]

在均匀spin Fourier展开中，log P的二阶系数只出现在三条真实pair上，而且均为

\[
 J=-A/8>0.
 \tag{7}
\]

其余二阶系数为零。

### 3.1 Laminar祖先强制的二阶系数等式

任意节点势g_S(N_S)对S内部的坐标置换对称。因此它对每一对{i,j}包含于S的二阶Fourier系数都相同，记为beta_{S,2}。unary没有二阶贡献，所以

\[
 \widehat{\log q}(\{i,j\})
 =\sum_{S\supseteq\{i,j\}}\beta_{S,2}.
 \tag{8}
\]

假设一个大小至少2的节点S包含真实pair的一端i，却不包含其另一端j；取任意k属于S且k不等于i。每个同时包含i,j的节点与S相交，又包含S外的j，laminar性强迫它包含整个S，因而也包含k。反方向对同时包含k,j的节点同理。因此

\[
 \{T:i,j\in T\}=\{T:k,j\in T\},
 \qquad
 \widehat{\log q}(\{i,j\})
 =\widehat{\log q}(\{k,j\}).
 \tag{9}
\]

目标中第一条是非零matching系数J，第二条是非matching系数0。

令e=||log q−log P_epsilon||_infinity。任一Fourier系数的误差不超过e。于是若上述切开pair的S存在，(9)给出J<=2e。因此

\[
 e<J/2
 \quad\Longrightarrow\quad
 \text{每个非单点节点都是完整真实pair的并。}
 \tag{10}
\]

该论证允许二阶势之间任意抵消，不要求每个节点的二阶系数非零。

### 3.2 剩下的pair并节点不能产生三重反对称项

在每对恰好一个1的八种配置上，令z_a=c_{2a-1}−c_{2a}属于{−1,1}。若所有非单点节点都是完整pair的并，其count在这八种配置上均为常数。任意RC的log q限制在此取向立方体上只能是常数加unary线性项：

\[
 \log q|_{\rm orientations}
 =b_0+b_1z_1+b_2z_2+b_3z_3.
 \tag{11}
\]

目标却由(6)给出常数+B z_1z_2z_3。对误差取三阶Walsh系数，其绝对值为B且不超过e，故e>=B。结合(10)，对任何无界普通laminar RC，

\[
 \boxed{\|\log q-\log P_\epsilon\|_\infty
 \ge \min\{J/2,B\}>0.}
 \tag{12}
\]

特别地，P_epsilon不能由任何这种RC精确表示。这不是“每种RC的最优分布都是均匀”的结论；切开真实pair的节点仍可部分降低KL。

## 4. 显式forward KL下界与有限拓扑闭包

令p_min=2^{-6}(1−epsilon)，D_q=D(P_epsilon||q)。Pinsker不等式给出

\[
 \max_c|q(c)-P_\epsilon(c)|
 \le {\rm TV}(P_\epsilon,q)
 \le\sqrt{D_q/2}.
 \tag{13}
\]

若D_q<p_min^2/2，则所有q(c)>p_min/2。log函数在该区域的导数不超过2/p_min，故

\[
 \|\log q-\log P_\epsilon\|_\infty
 \le \frac{\sqrt{2D_q}}{p_{\min}}.
 \tag{14}
\]

取c_0=min{1,J/2,B}。假若D_q<p_min^2 c_0^2/2，则(14)严格小于c_0，从而与(12)矛盾。因此

\[
 \boxed{
 \inf_{q\in{\rm RC}_6}D(P_\epsilon\|q)
 \ge \delta_{\rm exp}(\epsilon)
 :=\frac{p_{\min}^2}{2}
   \min\{1,J/2,B\}^{2}>0.
 }
 \tag{15}
\]

对epsilon=9/10，有p_min=1/640、J/2=log(100/19)/32、B=(log19)/2，且J/2为最小项。因而精确符号常数是

\[
 \boxed{
 \delta_{\rm exp}(9/10)
 =\frac{\log^2(100/19)}{838860800}.
 }
 \tag{16}
\]

这是约10^{-9}量级的保守解析常数，不是已冻结T包约0.049的有限box数值下界。这里不提供求积、拟合或浮点严格区间证书；(16)本身是精确表达式。

正下界也可由有限维闭包单独证明。六个标签的任意laminar族可补全为一棵含全根的满二叉层级，将新增节点势设零即可保持原分布。这样的树共有(2·6−3)!!=945种。对固定树tau，把常数项计入特征，所有log q都位于R^64中一个闭线性子空间V_tau。若存在q_n使KL趋零，则q_n趋于fullsupport的P，因而log q_n趋于log P。有限种树保证存在固定树子列，遂log P属于V_tau，与(12)矛盾。

此闭包论证中的节点势无界。T包枚举相同945个拓扑但施加了固定参数box；它的数值lower/upper和本节无界常数不是同一个对象，也不能相互替代。

## 5. 条件化闭合与任意维数的总KL下界

令A={1,...,6}，I={7,...,D}。对任意fullsupport Q_D属于普通RC_D，固定非活跃配置z=C_I。原节点势变为

\[
 g_S(N_S)
 \longmapsto
 g_S(N_{S\cap A}+k_S(z)),\qquad
 k_S(z)=\sum_{i\in S\cap I}z_i.
 \tag{17}
\]

原laminar集合族与A相交后仍laminar。重复交集对应的势相加，空交集给常数，单点交集可并入unary。于是

\[
 Q_D(C_A\mid C_I=z)\in{\rm RC}_6.
 \tag{18}
\]

条件化后的势可能超出原有限box，所以这里必须使用无界下界(15)，不能使用T的box数值。

由KL链式以及(3)，

\[
 \begin{aligned}
 D(P_{\epsilon,D}\|Q_D)
 &=D(U_I\|Q_I)
   +E_{z\sim U_I}
     D(P_\epsilon\|Q_D(\,\cdot\mid z))\\
 &\ge\delta_{\rm exp}(\epsilon).
 \end{aligned}
 \tag{19}
\]

因此，任意D>=6、任意层级结构、任意节点数和无界有限势的普通laminar RC，都不能把该完整joint KL降到零；下界与D无关。这是特定表示族的障碍，不是训练算法失败、所有推断的困难或指数计算下界。

若epsilon允许带符号且epsilon_min<=|epsilon|<=a<1，坐标重排和pair内部方向交换保持RC族。由(15)可给出对全部这种幅度、活跃坐标及matching统一的常数

\[
 \delta_{\rm class}
 =\frac{(1-a)^2}{2\cdot64^2}
 \min\left\{1,
 \frac{-\log(1-\epsilon_{\min}^2)}{32},
 \operatorname{atanh}\epsilon_{\min}\right\}^{2}>0.
 \tag{20}
\]

## 6. 固定活跃块LOO查询的严格正下界

在六维上定义每次单bit查询的额外log loss

\[
 L_{\rm LOO}(P,q)=
 \frac16\sum_{j=1}^{6}
 E_{C_{-j}\sim P}
 D\!\left(P(C_j\mid C_{-j})
 \,\middle\|\,q(C_j\mid C_{-j})\right).
 \tag{21}
\]

所有六种条件预测必须来自同一个归一化q。我们证明

\[
 \delta_{\rm query}(\epsilon)
 :=\inf_{q\in{\rm RC}_6}L_{\rm LOO}(P_\epsilon,q)>0,
 \tag{22}
\]

但不给它赋予T的query endpoint值，也不声称它等于delta_exp。

若(22)为零，存在q_n使(21)趋零。P_epsilon的每个上下文都有正概率，且这里只有有限个上下文，所以每个Bernoulli conditional KL都趋零。于是每条超立方体边的条件概率与odds都趋于P的值，即

\[
 \frac{q_n(c_j=1,c_{-j})}{q_n(c_j=0,c_{-j})}
 \longrightarrow
 \frac{P_\epsilon(c_j=1,c_{-j})}
      {P_\epsilon(c_j=0,c_{-j})}.
 \tag{23}
\]

连接任意配置与固定参考配置的有限边路径，得到所有q_n(c)/q_n(c_ref)趋于对应P比例；归一化再给出q_n趋于P。因为P fullsupport，最终所有q_n有共同正下界，故joint KL也趋零，与(15)矛盾。于是(22)成立。

现在在D维，预先固定目标J均匀取前六维，观测其余D−1个bit，包括所有非活跃位。对每个非活跃配置z，(18)及(22)成立；对z平均得到

\[
 \inf_{Q_D\in{\rm RC}_D}
 \frac16\sum_{j=1}^{6}
 E_{P_{\epsilon,D}}
 D\!\left(P_{\epsilon,D}(C_j\mid C_{-j})
 \,\middle\|\,Q_D(C_j\mid C_{-j})\right)
 \ge\delta_{\rm query}(\epsilon)>0.
 \tag{24}
\]

目标bit尚未知，故完整count也尚未知；没有向查询器额外提供真N。

### 查询和表示结论的适用边界

- 若目标均匀取全部D维，本证明只提供(6/D)delta_query下界。实际上Q_D=q_6乘以U_{D-6}让非活跃目标的额外loss为零，故该合同下不能主张维数统一的正查询下界。
- 若非活跃变量隐藏并被边际化，Q_A是条件RC分布的混合，未必仍属于RC_6。本证明依赖观测并条件化，不覆盖这种潜变量混合。
- 不覆盖节点相互交叉的任意factor graph、RC混合、任意电路、query专用互不相容的预测器或任意新编码。
- (24)的工作负载外生固定为前六维。面对未知活跃集合的大族，不能把“目标均匀取真实活跃块”当成免费已知任务；第8节只用前六维活跃子族形成该固定工作负载下的minimax障碍。
- 正的无界query gap由解析证明给出，未在本包计算其数值。T只报告共同joint-fit终点上的query风险，不能当作query目标的全族最小值。

## 7. 未知活跃坐标与matching的一个构造性学习器

这里改变学习任务：D>=6，六个活跃坐标及其三对matching未知；r=3和差分模板已知。将每对按较小坐标在前定向，定义对应h_H。教师是

\[
 P_{H,\epsilon}(c)=2^{-D}[1+\epsilon h_H(c)],
 \qquad
 0<\epsilon_{\min}\le|\epsilon|\le a<1.
 \tag{25}
\]

epsilon_min和a已知，epsilon可正可负。pair排序不改变h；改变pair方向的总符号可吸收到epsilon。任意坐标bit flip不在这个族内：只翻pair的一端会把差分模板变成和式，不能只靠一个epsilon符号吸收。

学习器收到两个独立的完整向量iid训练split，大小为n_1、n_2；不收到活跃块或matching。该构造只作可学习性上界，不声称样本或计算最优。

### 7.1 第一split：三阶矩恢复结构

对全部M_D=binom(D,3)个不同坐标三元组T，计算

\[
 \widehat m_T=\frac1{n_1}\sum_{\ell=1}^{n_1}
 \prod_{i\in T}Y_i^{(\ell)}.
 \tag{26}
\]

由(4)的Fourier展开，恰有八个三元组具有非零总体矩：它们从每条真实pair各取一个端点，矩为加减epsilon/8。其余三阶矩均为零。

固定严格阈值，选择

\[
 \widehat{\mathcal T}
 =\{T:|\widehat m_T|>\epsilon_{\min}/16\}.
 \tag{27}
\]

每个被平均变量位于[−1,1]。Hoeffding与union bound给出

\[
 \Pr\left(
 \max_T|\widehat m_T-m_T|
 \ge\epsilon_{\min}/16
 \right)
 \le
 2\binom D3
 \exp\left(-\frac{n_1\epsilon_{\min}^2}{512}\right).
 \tag{28}
\]

在补事件上，所有零矩都低于阈值，所有非零矩严格高于阈值，所以恰好恢复八个真实transversal三元组。

恢复算法取这些三元组的坐标并集，并验证：

1. 并集恰好有六个坐标，且有八个不同三元组；
2. 六个坐标中，“从未共同出现在任何已选三元组”的配对关系构成perfect matching；
3. 已选集合恰好是该matching的全部八个transversal。

真实超图中，真实pair的两端从不共现，而任何不同pair的两端都共同出现在两个transversal中，因此matching唯一。这三个验证通过则取该matching；失败则直接输出U_D。在(28)补事件上，恢复必然正确。坏事件上可能验证出错误matching，风险界会覆盖它。

### 7.2 第二split：幅度估计与clip

若结构验证通过，以canonical方向定义恢复的h_hat，使用独立第二split：

\[
 \widetilde\epsilon=
 \frac8{n_2}\sum_{\ell=1}^{n_2}
 h_{\widehat H}(C^{(\ell)}),\qquad
 \widehat\epsilon=
 \operatorname{clip}_{[-a,a]}(\widetilde\epsilon).
 \tag{29}
\]

输出Q_hat=2^{-D}(1+epsilon_hat h_hat)。epsilon_hat可以为零，不要求再次满足幅度下界；输出仍归一化且fullsupport。

条件于第一split正确恢复结构，第二split仍来自同一个真分布且独立。对canonical真h，

\[
 E h=\epsilon/8,\qquad
 E h^2=1/8,\qquad
 \operatorname{Var}(\widetilde\epsilon)
 =\frac{8-\epsilon^2}{n_2}\le\frac8{n_2}.
 \tag{30}
\]

clip到含真epsilon的区间不会增加平方误差，所以条件MSE也不超过8/n_2。

### 7.3 精确KL与总体风险

当结构正确时，h=0区域上的P与Q相同；h=1及h=−1区域的总质量均由Bernoulli幅度控制。因此

\[
 D(P_{H,\epsilon}\|P_{H,\widehat\epsilon})
 =\frac18
 d_{\rm Ber}\left(
 \frac{1+\epsilon}{2}
 \,\middle\|\,
 \frac{1+\widehat\epsilon}{2}\right).
 \tag{31}
\]

Bernoulli KL不超过相应chi-square divergence，给出

\[
 d_{\rm Ber}\left(
 \frac{1+\epsilon}{2}
 \,\middle\|\,
 \frac{1+\widehat\epsilon}{2}\right)
 \le
 \frac{(\epsilon-\widehat\epsilon)^2}
      {1-\widehat\epsilon^2}
 \le\frac{(\epsilon-\widehat\epsilon)^2}{1-a^2}.
 \tag{32}
\]

正确结构事件上的条件expected joint KL不超过1/[n_2(1−a^2)]。若结构错误，任何通过验证的输出及uniform fallback都满足逐配置界

\[
 (1-a)2^{-D}\le Q_{\rm hat}(c)\le(1+a)2^{-D},
\]

真P也满足相同界，所以无条件有

\[
 D(P\|Q_{\rm hat})\le
 M_a:=\log\frac{1+a}{1-a}.
 \tag{33}
\]

结合(28)，对所有未知活跃集合、matching及允许的epsilon，

\[
 \boxed{
 \sup_{H,\epsilon} E D(P_{H,\epsilon}\|Q_{\rm hat})
 \le
 \frac1{n_2(1-a^2)}
 +2M_a\binom D3
  \exp\left(-\frac{n_1\epsilon_{\min}^2}{512}\right).
 }
 \tag{34}
\]

期望同时覆盖两个训练split。它是完整D维总KL上界，不是每维KL，也不是仅parameter一致性。

例如取

\[
 n_1=
 \left\lceil\frac{512}{\epsilon_{\min}^2}
       [4\log D+\log(1+M_a)]\right\rceil,
 \qquad
 n_2=\lceil\log D\rceil,
 \tag{35}
\]

并用binom(D,3)<=D^3/6，得到

\[
 \sup_{H,\epsilon}E D(P_{H,\epsilon}\|Q_{\rm hat})
 \le\frac1{(1-a^2)\log D}+\frac1{3D}
 \longrightarrow0.
 \tag{36}
\]

样本数为O(log D)，其中epsilon_min和a固定。朴素三阶矩累加算术量为O(n_1 D^3)，第二split只读已选六列时为O(n_2)；若必须顺序读取所有完整向量，另计O(n_2 D)输入成本。这是有效多项式构造，不是计算或样本下界，也不是具有普遍未知模型适应性的学习保证。

## 8. 可以据此主张的分离与冻结边界

对(25)的已知模板族，普通laminar RC有统一的总joint KL下界delta_class>0，而第7节在不知道活跃坐标及matching的情况下给出总joint KL趋零的显式学习器。故这里成立的是：

> 一个可从有限样本有效学习的稀疏高阶分布族，仍可能在所有普通laminar RC结构中有维数不消失的表示误差。

这不意味着任意学习器、任意推断或所有结构先验都困难，也不声称Fourier结构恢复是新方法。正面学习器可以输出RC族之外的已知模板分布，因此没有与负面定理矛盾。

对query，固定目标工作负载为“前六维均匀目标、观测其他全部位”。大教师族包含活跃集合恰为前六维、固定matching和固定epsilon的子族，因此任何RC预测程序在该子族上仍受(24)限制；这足以形成该外生工作负载下的minimax表示障碍。不能扩大成大教师族内每个未知support的点态查询下界。

第7节学习器的查询风险也可由joint KL控制：对任一目标j，KL链式给出其conditional KL不超过joint KL，对预先固定的目标分布取平均仍然成立。因此(34)同样给出此固定查询合同下趋零的上界；这里没有使用真活跃集合来选择查询目标。

### 与已冻结证据的关系及审查边界

本文件的依赖是THEORY_DESIGN中的普通RC定义与高阶见证；为了可独立审查，关键证明已在本文重写。新增内容是任意维数条件化嵌入、显式无界KL常数、LOO条件风险的正下界，以及未知活跃坐标与matching的模板学习上界。

- T的945个population拟合属于固定box数值证据；本文(15)、(19)和(22)不从这些拟合外推。
- T的最小joint-fit query endpoint不是本文delta_query的数值。
- 没有新增拟合、teacher样本、随机bank、优化重启、结构搜索实验或成功门。
- 全count和pair失明不等于所有诊断失明；三阶矩正面构造明确说明了其边界。
- 若更换差分模板、允许任意bit编码、未知阶数、隐变量边际化、混合RC或交叉节点，这些定理需要重新核验。
