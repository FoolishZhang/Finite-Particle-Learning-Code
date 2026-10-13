# 真实共享银行的有限网格ERM：原目标学得伪相关，同Q边际化恢复真值

新研究基准 `33f2d1c5378e269563a96cbfdd6eb0b2c32d9a7b`。本页只作解析推导，不生成标签、prior银行或新积分。模型保持已知两位active块，新增n位是真实观察上的独立公平因子；不扩为未知active位置或全维自由协方差。

## 1. 有限数据与真实global-shared银行

固定a>0、0<R<1和有限网格 $\mathcal G\subset[-R,R]$，包含 $0,\pm R$，其大小为M。教师完全独立，全部D=n+2位观察为独立公平bits。对N个完整标签 $Y_j\in\{-1,1\}^{n+2}$，生成独立于数据的B个完整prior银行，每银行K个标准Gaussian向量 $Z_{bk}$。所有标签和候选rho共用这些银行。候选logits为

\[
\eta_{bk}(\rho)=a\bigl(Z_{bk1},\ \rho Z_{bk1}+\sqrt{1-\rho^2}Z_{bk2},\ Z_{bk3},\ldots,Z_{bk,n+2}\bigr).
\tag{SE1}
\]

定义原始bank损失及实际有限目标

\[
L_\rho(Y,Z_b)=-\log\left\{\frac1K\sum_{k=1}^K\prod_i\sigma(Y_i\eta_{bki}(\rho))\right\},\quad
\widehat F^{\rm raw}_{N,B,K}(\rho)=\frac1{NB}\sum_{j,b}L_\rho(Y_j,Z_b).
\tag{SE2}
\]

这里NB个交叉项不是独立样本。尤其对每条实际标签，nuisance项为

\[
\frac a2Y_{j,\rm noise}^{\top}Z_{bk,\rm noise}
-\sum_{i>2}\log[2\cosh(aZ_{bki}/2)].
\tag{SE3}
\]

不能将所有标签符号吸收后，给不同j复用同一个canonical nuisance和；那会改变共享银行经验目标的联合分布。每标签独立银行是另一种合法算法，但不是本页合同。

同Q nuisance marginalization在每个bank的似然均值内部精确积分n个无关位：

\[
L^{\rm RB}_\rho(Y,Z_b)=n\log2-
\log\left\{\frac1K\sum_k\sigma(Y_1\eta_{bk1})\sigma(Y_2\eta_{bk2})\right\}.
\tag{SE4}
\]

它使用相同标签和相同银行的active部分，且每个候选的积分观察律Q完全不变。RB改变有限粒子评分，不是原loss的无偏控制变量；固定K时仍可能有active偏差。两种算法都在同一有限网格取实际经验最小值，不用Price积分估计器代替训练目标。

## 2. 与n无关的原目标对比包络

以rho=0为参数无关基准，令 $f^t_\rho=L^t_\rho-L^t_0$，其中 $t\in\{\mathrm{raw},\mathrm{RB}\}$。减去同一个基准不改变各方法的ERM。

log-sigmoid的导数绝对值不超过1，log-sum-exp对输入的无穷范数为1-Lipschitz。SE1只改变active第二个logit。因此逐标签、逐银行、逐rho有

\[
|f^t_\rho(Y,Z_b)|\le H_b:=a c_R\max_{k\le K}\|(Z_{bk1},Z_{bk2})\|,
\quad c_R^2=2(1-\sqrt{1-R^2}).
\tag{SE5}
\]

因为 $\rho^2+(\sqrt{1-\rho^2}-1)^2\le c_R^2$，这个界对整个网格和两种方法同时成立，且不含n。二维标准Gaussian范数的尾给

\[
P(H_b>h)\le K\exp\{-h^2/(2s^2)\},\qquad s=a c_R.
\tag{SE6}
\]

## 3. 两层有限样本集中，不假设NB独立

**引理。** 令0<δ<1，并取

\[
h=s\sqrt{2\log(3BK/\delta)},\qquad
L=\log(12M/\delta),
\]
\[
e_{N,B,K,\delta}=h\left(\sqrt{2L/N}+\sqrt{2L/B}\right)
+K\left(h+\frac{s^2}{h}\right)e^{-h^2/(2s^2)}.
\tag{SE7}
\]

则以至少1−δ的概率，raw与RB同时满足

\[
\sup_{\rho\in\mathcal G}
|\widehat\Delta^t(\rho)-\Delta^t(\rho)|\le e_{N,B,K,\delta},
\quad\widehat\Delta^t=\widehat F^t(\rho)-\widehat F^t(0).
\tag{SE8}
\]

**证明。** 仅在分析中定义截断函数 $f^{t,h}_\rho=f^t_\rho\mathbf1\{H\le h\}$，真实训练算法不作此截断。全部B个实际银行均满足H≤h的失败概率由SE6和union bound不超过δ/3。

对截断后的交叉平均，先条件于B个银行。关于N个iid标签的平均与其条件均值之差，应用范围 $[-h,h]$ 的双侧Hoeffding界。再对 $g^t_\rho(Z)=E_Y f^{t,h}_\rho(Y,Z)$ 的B个iid银行平均应用相同界。两层分别对2M个函数union；选择L如SE7，每层失败概率至多 $4M e^{-L}=\delta/3$。条件于银行的标签层概率上界不依赖具体银行，积分后仍有效。两层事件可依赖，不要求它们独立。

截断的总体偏差统一不超过

\[
E[H\mathbf1\{H>h\}]
=hP(H>h)+\int_h^\infty P(H>u)du
\le K\left(h+\frac{s^2}{h}\right)e^{-h^2/(2s^2)}.
\]

合并两层偏差、截断尾和三个失败事件即得SE8。注意两层分别产生 $N^{-1/2}$ 与 $B^{-1/2}$，不是 $(NB)^{-1/2}$。□

## 4. 多项式数据与银行下，raw网格ERM选错端点

本节引用已封存 [POLYNOMIAL_K_THEORY](../无关维度与固定噪声总体偏好_20261001/POLYNOMIAL_K_THEORY.md) 的PK27–PK32。该原证明与本页来自同一执行者；这里是应用已有定理，**不称本页对原证明作了新的独立外审**。原独立审读见 POLYNOMIAL_K_REVIEW (historical reference not distributed)。原理论SHA为 `e88b5e4de6c4ac4210708bb366d9f299a124b6ec118e199461a56fda04d08d9d`，审读SHA为 `0cc2b239e491648011ddbafcf99eda76c21dd8e99bd4948b8c7a0703513c0102`。

固定η>0，取

\[
K_n=n+2,\qquad N_n=B_n=\lceil n^{1+\eta}\rceil,\qquad\delta_n=n^{-2},\quad n\ge2.
\tag{SE9}
\]

SE7给 $e_n=O(\log n/n^{(1+\eta)/2})$。旧定理的 $J_{K_n}=E\max_{k\le K_n}G_k\sim\sqrt{2\log n}$，因此

\[
e_n=o(J_{K_n}/\sqrt n).
\tag{SE10}
\]

独立教师下，raw总体对比乘 $\sqrt{nv}/J_{K_n}$（其中 $v=\operatorname{Var}\log\sigma(aZ)$）后，在整个rho区间一致趋于

\[
V_0(\rho)=-\frac{a^2}{4}\int_0^\rho r_a(u)du,
\quad r_a(u)=E_u[\tanh(aX_1/2)\tanh(aX_2/2)].
\tag{SE11}
\]

V严格凹且偶对称，最小点恰为±R。固定有限网格中，所有其它点与端点的极限gap具有正最小值。因此对足够大n，raw的总体网格gap至少为正常数乘 $J_{K_n}/\sqrt n$。SE8与SE10使实际经验网格ERM无法选择其它点：

\[
P\{\widehat\rho^{\rm raw}_n\in\{-R,R\}\}\longrightarrow1.
\tag{SE12}
\]

这是原经验log-mean目标的精确网格ERM，不是Price目标或总体oracle选择。若使用近似算术/优化，另须证明其目标残差小于上述gap尺度；本文理论不把固定浮点容差或某个执行器直接认证为渐近精确。

## 5. 同数据、同银行的RB网格ERM恢复真值

对固定active标签y与候选rho，令单粒子active似然 $W_k\in(0,1)$，均值为 $q_\rho(y)>0$。SLLN给 $\overline W_K\to q_\rho(y)$ 几乎处处。又由两次Jensen，

\[
0\le-\log\overline W_K\le\frac1K\sum_k[-\log W_k],\quad
E[(-\log\overline W_K)^2]\le E[(-\log W_1)^2]<\infty.
\tag{SE13}
\]

最后的二阶矩由 $-\log W_1\le2\log2+a(|X_1|+|X_2|)$ 及Gaussian矩控制。故这些损失一致可积，期望也收敛到 $-\log q_\rho(y)$。标签只有四种、网格有限，所以RB总体对比在网格上一致收敛到精确积分模型的交叉熵差。

rho=0是真值。Price公式给 $r_a'(\rho)>0$，所以其它网格点的观察分布不同，其完整观察KL严格正。精确交叉熵在有限网格上因此有唯一最小点0和固定正gap。SE8中的RB经验误差趋零，且它与raw是同一个联合概率事件，于是

\[
P\{\widehat\rho^{\rm RB}_n=0\}\longrightarrow1.
\tag{SE14}
\]

共享数据、银行或跨n前缀不会使两个结论要求额外独立性。它们只用SE8已经对raw/RB联合union过的一次事件。

完整观察KL只依赖active块，且在紧rho区间上有界。因此概率收敛可升级为程序期望风险：

\[
E D(P\Vert Q_{\widehat\rho^{\rm raw}_n})\to
-\tfrac12\log(1-r_a(R)^2)>0,
\qquad
E D(P\Vert Q_{\widehat\rho^{\rm RB}_n})\to0.
\tag{SE15}
\]

raw两个端点的风险相同，故其符号随机性不影响第一极限。这里K增长是必要合同的一部分；固定K2/4的实验中，RB只保证去掉padding效应，不由本节保证其有限K目标等于精确似然。

## 6. 实验统计的解析保证与不同置信家族

本节限实际实验a=1、R=3/4和null教师。令 $\mathcal R(\rho)=-\frac12\log(1-r_1(\rho)^2)$。Price及 $\operatorname{sech}^2x\ge e^{-x^2}$ 给

\[
\frac16\le r_1'(\rho)\le\frac14,
\quad |r_1(\rho)|\le|\rho|/4.
\tag{SE16}
\]

下界来自二维Gaussian二次型积分
$\frac14[(3/2)^2-\rho^2/4]^{-1/2}\ge1/6$。因此对 $0\le v\le u\le R$，

\[
\mathcal R(u)-\mathcal R(v)\ge\frac{u^2-v^2}{72},\qquad
\mathcal R(\rho)\le\frac{\rho^2}{32(1-R^2/16)}.
\tag{SE17}
\]

在实验网格 $\rho_i=3i/20$（i=−5,…,5）上，若 $i_{\rm raw}^2-i_{\rm RB}^2\ge4$，就解析保证真实积分KL改善严格大于.001。对16个独立训练stream，以这个整数事件的次数x作单侧精确二项检验：p值为 $\sum_{j=x}^{16}\binom{16}{j}/2^{16}$，alpha=.05。这检验的是改善事件的概率超过1/2，**不是平均风险差超过.001**。参考积分风险仍完整报告，但不进入主事件。

另一个家族含高预算K2/K4各一个raw端点事件。若其概率为p，解析 $\mathcal R(\pm R)\ge1/128$ 给 $E\mathcal R(\widehat\rho^{raw})\ge p/128$。各单侧CP下界用alpha=1/40，共同覆盖至少.95；不把这个家族与前一个sign检验拼成全报告共同95%。

CP下界可用40位dyadic二分全整数实现。对x>0，寻找 $p_L=u/2^{40}$，使

\[
40\sum_{j=x}^{16}\binom{16}{j}u^j(2^{40}-u)^{16-j}\le2^{640},
\tag{SE18}
\]

并保存下一个dyadic点不满足该式的证据；x=0取0。这个向下取整的CP下界保守，数值逆CDF不进入证书。

表格似然基线只依赖N个active标签中的same次数s，null下 $s\sim\operatorname{Bin}(N,1/2)$。即使表格概率只有数值资格，它给出的固定选择规则 $i_m(s)$ 也满足逐输出解析上界。因为SE17在R=3/4时等于 $9i^2/12350$，其真实程序期望风险有精确有理上界

\[
E\mathcal R(\widehat\rho^{\rm tableMLE})
\le\frac{9}{12350\,2^N}\sum_{s=0}^N\binom Ns i_m(s)^2.
\tag{SE19}
\]

所以原两个CP事件还确定性推出raw期望风险相对该表格程序的下界 $p_L/128-\text{SE19}$；不是另一个置信家族。它不依赖参考求积精确性，也不把表驱动程序叫作精确连续MLE。所有抽样保证均以训练stream理想iid为条件；不另声称PRNG分布正确性的证明。

## 7. 范围与实现边界

理论的N/B/K共同增长路径与固定K2/4的有限实验是不同量词。SE7是充分上界，当前预算可能给约数十分之一nat的空界，不能删去或用它认证实测细小评分差。有限N/B/n/K前缀彼此依赖，统计独立单位是训练stream。

候选只改变已知active相关，均值、边际尺度和nuisance参数固定；没有连续全局优化、未知位置学习或自由协方差结论。raw优化随机K预测器的期望log风险，本文评价积分模型Q；二者的风险合同不可混淆。旧poly-K、源码和全部历史冻结均不回写。
