# 有限标签下先平均银行的active目标：事后诊断身份

本页在观察到本轮高预算K2的RB参考风险仍接近raw后提出。它是事后机制诊断，不改原学习freeze、paired主门或CP家族，不宣称独立确认。0新Gaussian积分、标签、prior银行或训练。

## 1. 条件于数据的目标，只剩经验pair矩

固定active两位、零均值Gaussian prior、尺度a=1及候选rho。令 $L_K(\rho)$ 为全正标签的active负log-mean likelihood对K个iid prior粒子取期望。Gaussian整体符号翻转给全负标签同一个期望；只翻第二位则把相关rho变为−rho。因此同标签的两个模式共享 $L_K(\rho)$，异标签的两个模式共享 $L_K(-\rho)$。

给定N条真实active标签，令s为same次数，$t=2s/N-1$。不管四模式的经验边际是否平衡，先对全部银行取期望后，active经验目标都是

\[
F_{t,K}(\rho)=\frac{1+t}{2}L_K(\rho)+\frac{1-t}{2}L_K(-\rho).
\tag{AT1}
\]

这里t可取整个[-1,1]；它是经验自旋pair矩，不必处于候选Gaussian观察pair矩的可达范围。不能把t裁剪成候选可达相关，也不能把它误作潜在rho。

这是 $E_{\rm banks}[\widehat F_K(\rho)\mid Y_{1:N}]$ 的恒等式，不是 $E[\arg\min\widehat F\mid Y]$。有限B的共享银行目标仍依赖各条真实标签及实际prior；argmin与银行期望不能交换。本页没有将真实有限B训练替换为canonical银行。

## 2. 从旧非零教师表反演所有经验t

旧 `artifacts/weak_emission/NUMERIC/GRID.json` 在latent教师theta=.15下保存了active总体目标差 $D_*(\rho)=F_{r_*,K}(\rho)-F_{r_*,K}(0)$。其中 $r_*$ 必须取同变体的 `teacher_spin_moment`；它是观察pair矩，不能用latent .15替代。

若 $r_*\ne0$，定义

\[
E_K(\rho)=\frac{D_*(\rho)+D_*(-\rho)}2,\qquad
O_K(\rho)=\frac{D_*(\rho)-D_*(-\rho)}{2r_*}.
\tag{AT2}
\]

由AT1的两条线性方程直接解得

\[
\boxed{F_{t,K}(\rho)-F_{t,K}(0)=E_K(\rho)+tO_K(\rho).}
\tag{AT3}
\]

E为偶函数、O为奇函数；t=0的符号平局应保留为集合。它不需要t对应某个Gaussian教师参数，也不产生新的求积节点。

对本轮nuisance-RB，完整目标只是AT1加n log2，故AT3对其选择适用。旧表的n=0系数**不能代入n254 raw**；后者需要含nuisance粒子因子的另一套期望系数。本包不建立或计算那些系数。

## 3. 精确binomial权重与参考目标的限制

null教师下，$s\sim\operatorname{Bin}(N,1/2)$。冻结的数值系数表对每个s定义一个网格argmin集合 $\mathcal A(s)$。由全状态枚举可以精确按binomial概率汇总该表驱动proxy的行为，但这不使旧系数成为严格Gaussian期望证书，更不认证真实B∞连续最优点。

为了不挑选最好平局，参考期望风险报告允许argmin集合中所有选择的下/上端：

\[
\sum_s\frac{\binom Ns}{2^N}\min_{i\in\mathcal A(s)}R_i^{\rm ref},
\qquad
\sum_s\frac{\binom Ns}{2^N}\max_{i\in\mathcal A(s)}R_i^{\rm ref}.
\tag{AT4}
\]

参考浮点常数转为精确有理数后再加权，故“精确”仅指给定参考表与binomial权重的算术。若平局只有±rho，真实null风险相同；不强迫一个符号充当唯一target。不同幅度的平局也完整保留，不按真实风险择优。

在rho=3i/20的当前11格上，任一输出仍有解析真风险上界 $\mathcal R(\rho_i)\le9i^2/12350$。因此对所有选取argmin集合成员的规则均有

\[
E\mathcal R\le\frac{9}{12350\,2^N}
\sum_{s=0}^N\binom Ns\max_{i\in\mathcal A(s)}i^2.
\tag{AT5}
\]

它不依赖参考积分准确性，但只约束这套已冻结表定义的规则，不把其选点认证为真实无限银行目标的精确极小点。与原表格似然基线的比较也是诊断；不增加置信家族或修改原成功/失败。

## 4. 能解释与不能解释的内容

按实际same-count查表，比较全N/K/stream的target集合与原三个B下RB输出，可以区分“有限标签已经使条件期望目标偏离真值”与“有限银行进一步改变了所选网格点”。风险差不一定同号；有限银行也可能偶然把结果移近真值，不能将差值自动称为因果贡献比例。

本包保留全部s=0,…,N、全部三变体、全部实际count和B前缀。N/B与同一stream内的比较相关，无新CI、显著性或独立确认。原raw/RB训练、模型输出、paired主门、CP下界、同Q渐近定理均不回写。
