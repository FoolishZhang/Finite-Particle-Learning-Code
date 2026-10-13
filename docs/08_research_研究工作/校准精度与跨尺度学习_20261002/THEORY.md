# 有限校准误差如何传递到精确似然、参数与风险

日期：2026-10-02。本文为新的跨尺度包提供解析接口，零 RNG、零模型积分、零训练。拟议有限设置为 $a=1,R=.75$、$n\in\{62,254\}$、$K\in\{64,256\}$；下述有限命题对每个固定 $n\ge0,K\ge2,0<R<1$ 成立，不预设四格具有相同校准、曲率或 Monte Carlo 精度。旧科学包和稿件保持不变。

前提是已知两位相关块、固定发射模型、标准 iid 等权 prior 粒子及完整共享 bank。沿用已证明的 有限 n/K 校准身份 (historical reference not distributed)：对固定完整数据的经验自旋乘积 $t\in[-1,1]$，raw 条件 bank 期望为 $U(\rho)+tV(\rho)$，理想加性校准满足 $g'=-U'-rV'$，其中 $b=-V'>0$。一般一致评分与误差传递是经典思想；本文件给出这个具体目标的常数和边界处理，不宣称新的通用评分方法。

## 1. 不同目标之间存在一个正斜率比

令 $I=[-R,R]$，$r$ 是本 Gaussian-logistic 两位观察律的相关矩。它为奇函数，在 I 上光滑且严格递增，$r'>0$。定义

$$
\begin{gathered}
r_R=r(R)<1,\quad m_r=\min_I r'>0,\quad M_r=\max_I r',\\
b_* =\min_I b>0,\quad b^*=\max_I b,\qquad
q(t)=r^{-1}\bigl(\operatorname{clip}(t,[-r_R,r_R])\bigr).
\end{gathered}
\tag{C.1}
$$

精确经验负 log-likelihood 去掉参数无关常数后写为

$$
\ell_t(\rho)=-\frac{1+t}{2}\log(1+r(\rho))
-\frac{1-t}{2}\log(1-r(\rho)),
\qquad
\ell_t'(\rho)=\frac{r'(\rho)[r(\rho)-t]}{1-r(\rho)^2}.
\tag{C.2}
$$

这里 $t=\pm1$ 也合法；零权重项按零处理。令 $J_t=U+tV+g$，则

$$
J_t'(\rho)=b(\rho)[r(\rho)-t]
=\lambda(\rho)\ell_t'(\rho),\qquad
\lambda(\rho)=\frac{b(\rho)[1-r(\rho)^2]}{r'(\rho)}>0.
\tag{C.3}
$$

记 $\lambda_* =\min_I\lambda$、$\lambda^*=\max_I\lambda$。可用保守常数

$$
\lambda_*\ge\frac{b_*(1-r_R^2)}{M_r},\qquad
\lambda^*\le\frac{b^*}{m_r}.
\tag{C.4}
$$

这些常数的存在性不等于已获得当前四个 n/K 设置下的严格数值下界；若要将下文界当作数值认证，必须另外认证所用下界及评分误差。

**命题1（目标超额的双向比较，含 clipped MLE）。** 对每个 $t\in[-1,1]$ 和 $\rho\in I$，

$$
\lambda_*[\ell_t(\rho)-\ell_t(q)]
\le J_t(\rho)-J_t(q)
\le\lambda^*[\ell_t(\rho)-\ell_t(q)],\qquad q=q(t).
\tag{C.5}
$$

**证明。** 两个导数均在 q 左侧非正、右侧非负。如果 q 为右端点，则二者在 I 内非正；左端点情形相反。从 q 向 $\rho$ 的有向积分中，$\ell_t'$ 始终贡献非负增量，所以可在积分内用 $\lambda_*,\lambda^*$ 分别夹住 $\lambda$。这不依赖 q 为驻点或目标凸。□

两个目标具有同一个唯一最小点，但不是相等函数；$\lambda$ 通常不是常数。还可直接从单调性得到

$$
\begin{aligned}
J_t(\rho)-J_t(q)&\ge\frac{b_*m_r}{2}|\rho-q|^2,\\
\ell_t(\rho)-\ell_t(q)&\ge\frac12|r(\rho)-r(q)|^2
\ge\frac{m_r^2}{2}|\rho-q|^2.
\end{aligned}
\tag{C.6}
$$

第一式将 $|r(u)-t|\ge|r(u)-r(q)|\ge m_r|u-q|$ 沿正确方向积分；第二式在观察矩坐标 $v=r(\rho)$ 下积分 $(v-t)/(1-v^2)$，利用 $1/(1-v^2)\ge1$。clipped 情形下 t 位于可达区间之外，使有向积分更大。**这些是二次增长界，不是全域强凸性结论**：$J_t''=br'+b'(r-t)$ 的第二项没有固定符号。

## 2. 函数误差与近优误差到精确经验似然

固定实际完整数据，令 $\widehat F$ 为实际有限共享 bank 的 raw 经验目标，$F_t=U+tV$ 为其条件 bank 期望。用已部署校准 $\widehat g$，定义

$$
\widehat J=\widehat F+\widehat g=J_t+e_{MC}+e_g,\qquad
e_{MC}=\widehat F-F_t,\quad e_g=\widehat g-g.
\tag{C.7}
$$

各函数可以相差任意参数无关常数。对有界函数 $f$ 记 $\operatorname{osc}_I f=\sup_I f-\inf_I f$；这是与常数无关的误差量。假定输出满足真算术连续经验目标的近优条件

$$
\widehat J(\widehat\rho)\le\inf_{\rho\in I}\widehat J(\rho)+\eta.
\tag{C.8}
$$

设 $\operatorname{osc}e_{MC}\le\omega_{MC}$、$\operatorname{osc}e_g\le\omega_g$，并记 $E_0=\eta+\omega_{MC}+\omega_g$。则

$$
\boxed{
\ell_t(\widehat\rho)-\ell_t(q)\le\frac{E_0}{\lambda_*},\qquad
|\widehat\rho-q|\le\sqrt{\frac{2E_0}{b_*m_r}}.
}
\tag{C.9}
$$

**证明。** 将 q 代入 (C.8)，有 $J_t(\widehat\rho)-J_t(q)\le\eta+e_{MC}(q)-e_{MC}(\widehat\rho)+e_g(q)-e_g(\widehat\rho)\le E_0$。分别用 (C.5) 和 (C.6) 即得。□

特别地，若对某一共同常数锚点有全域函数误差 $\|e_{MC}\|_\infty\le\varepsilon_{MC}$，可用 $\omega_{MC}\le2\varepsilon_{MC}$；若另有 $\|e_g'\|_\infty\le\delta_g$，则 $\omega_g\le2R\delta_g$。校准在0归一化不是必要条件，不应让不同积分常数制造虚假的校准误差。

这里 $\eta$ 必须对应完整部署目标，包含确实已控制的网格、插值、点值和求解误差。只证明一个网格离散化项，并不等于证明全部 $\eta$；未认证的浮点误差仍需独立列出。$\varepsilon_{MC}$ 和 $\omega_g$ 是全域误差，不是单个结点SE、两个复本的距离或曲线光滑程度。

### 2.1 同时利用函数 MC 界与校准导数界

如果 $\operatorname{osc}e_{MC}\le\omega_{MC}$、$\|e_g'\|_\infty\le\delta_g$，则令 $S=\eta+\omega_{MC}$、$\gamma=b_*m_r$，还可用只沿选点路径累积的误差：

$$
\begin{gathered}
d:=|\widehat\rho-q|\le
D(S,\delta_g):=\frac{\delta_g+\sqrt{\delta_g^2+2\gamma S}}{\gamma},\\
\ell_t(\widehat\rho)-\ell_t(q)
\le\frac{S+\delta_gD(S,\delta_g)}{\lambda_*}.
\end{gathered}
\tag{C.10}
$$

证明为 $\gamma d^2/2\le S+\delta_g d$ 后解二次不等式，再用 (C.5)。该函数值论证的常数不应冒充下面利用一阶最优条件得到的更紧常数。

### 2.2 到真教师观察 KL 的条件传递

若教师 $\theta\in I$，记 $t_\star=r(\theta)$。投影非扩张性给 $|r(q)-t_\star|\le|t-t_\star|$。完整观察的 nuisance 项相同而抵消；在相关矩坐标对精确似然导数积分，得

$$
\mathrm{KL}(Q_\theta\|Q_\rho)
=\int_{t_\star}^{r(\rho)}\frac{v-t_\star}{1-v^2}\,dv
\le\frac{|r(\rho)-t_\star|^2}{2(1-r_R^2)}.
\tag{C.11}
$$

由 (C.6)、(C.9)，可得

$$
\mathrm{KL}(Q_\theta\|Q_{\widehat\rho})
\le\frac{\left\{\sqrt{2E_0/\lambda_*}+|t-t_\star|\right\}^2}{2(1-r_R^2)}
\le\frac{2E_0/\lambda_*+|t-t_\star|^2}{1-r_R^2}.
\tag{C.12}
$$

这是逐数据、逐误差事件的条件界。对于新 iid 完整观测，$t$ 是取值±1的 iid 乘积均值，可另用 $P(|t-t_\star|>s)\le2e^{-Ns^2/2}$；但这只控制数据项，不能替代对函数MC、校准和求解误差的认证。若各自另有误差事件，才可用 union bound 合并，不要求这些事件独立。当前小样本开发区间不会因 (C.12) 自动成为严格总体置信区间。

如果对全部随机输入，某个误差事件 $\mathcal E$ 满足 $P(\mathcal E^c)\le\delta_{fail}$，并且在其上 (C.8) 及 $E_0\le D_0$ 成立，其中 $D_0$ 是确定上界，则可得到真正的期望风险接口。由 (C.11) 可取统一最大风险上界 $R_{max}=2r_R^2/(1-r_R^2)$；iid 数据还给 $E|t-t_\star|^2=(1-t_\star^2)/N$。故

$$
E\,\mathrm{KL}(Q_\theta\|Q_{\widehat\rho})
\le\frac{2D_0/\lambda_*+(1-t_\star^2)/N}{1-r_R^2}
+R_{max}\delta_{fail}.
\tag{C.12a}
$$

证明先将期望按 $\mathcal E$ 与其补集拆开，在好事件上用 (C.12)，再以 $E[|t-t_\star|^2 1_{\mathcal E}]\le E|t-t_\star|^2$ 控制数据项；因此不要求误差事件和数据独立。若误差界逐实现成立可令 $\delta_{fail}=0$。只有高概率控制时不能悄然丢弃失败项；只有固定旧数据的条件梯度读数，也不能直接代入新 iid 数据的方差公式。这里没有为本次有限预算计算或认证 $D_0,\delta_{fail}$。

## 3. 导数误差与约束一阶条件

这部分假设实际完整扰动 $e=e_{MC}+e_g$ 可微，且 $\|e'\|_\infty\le\delta$。函数值小误差本身不推出这个条件；对近似多项式的函数值误差界也不能直接拿来认证其导数。

对区间约束定义一阶残差

$$
\mathcal S_I(f,x)=
\begin{cases}
|f'(x)|,&-R<x<R,\\
\max\{0,-f'(-R)\},&x=-R,\\
\max\{0,f'(R)\},&x=R.
\end{cases}
\tag{C.13}
$$

真实约束局部最小点具有残差0；仅有低函数值容差的求解器不自动满足一个规定的残差上界。若某输出满足 $\mathcal S_I(\widehat J,\widehat\rho)\le\kappa$，令 $c=(\delta+\kappa)/b_*$，则有

$$
\boxed{
|r(\widehat\rho)-r(q)|\le c,\quad
|\widehat\rho-q|\le c/m_r,\quad
\ell_t(\widehat\rho)-\ell_t(q)
\le\frac{c^2}{2(1-r_R^2)}.
}
\tag{C.14}
$$

**证明。** 若 $\widehat\rho=q$，结论直接成立。否则，假设 $\widehat\rho>q$；一阶残差条件在内点或右端点都给 $\widehat J'(\widehat\rho)\le\kappa$。故 $b(\widehat\rho)[r(\widehat\rho)-t]\le\delta+\kappa$。在该方向 $r(\widehat\rho)-t\ge r(\widehat\rho)-r(q)>0$。反方向同理，因此当输出不同于 q 时有

$$
|r(\widehat\rho)-t|\le c,\qquad
|r(\widehat\rho)-t|=|r(\widehat\rho)-r(q)|+|r(q)-t|.
\tag{C.15}
$$

在观察矩坐标积分 (C.2)，并用 $1/(1-v^2)\le1/(1-r_R^2)$，得

$$
\ell_t(\widehat\rho)-\ell_t(q)
\le\frac{|r(\widehat\rho)-t|^2-|r(q)-t|^2}{2(1-r_R^2)}
\le\frac{c^2}{2(1-r_R^2)}.
\tag{C.16}
$$

其余两项由 (C.15) 和 $r'\ge m_r$。□

这里覆盖 clipped MLE 的关键是**一阶条件加导数误差**，不是假定边界处精确似然的超额总是参数距离平方。一般边界有 $\ell_t'(q)\ne0$，任意邻近点的超额可能含线性项。进一步，若严格 clipped gap $|t-r(q)|>c$，(C.15) 排除所有不同于 q 的合格输出；此时输出必须仍在 q。边界的合法非零梯度不能被当成校准误差。

若 raw 的导数误差界为 $\delta_{MC}'$、校准导数误差为 $\delta_g$，可取 $\delta=\delta_{MC}'+\delta_g$。因此比较跨 n/K 的校准精度时，真正相关的是相对于 $b_*$ 的误差，而不是仅比较未归一化的 SE。

### 3.1 局部误差需要先有定位依据

如果上述导数界只在 $I_0=I\cap[q-r_0,q+r_0]$ 成立，只有当输出已知属于该邻域时，才能把 (C.14) 中的 $b_*,m_r,r_R$ 换成沿 $I_0$ 的对应界。单凭局部导数一致不能排除远处伪极小点。

一个充分定位条件由第2节直接给出。若全域 $E_0<\gamma r_0^2/2$，其中 $\gamma=b_*m_r$，则 (C.6) 排除全部距 q 至少 $r_0$ 的近优输出；之后才可使用局部导数误差和局部斜率。也可由其它独立全局定位结果建立该前提。不能用正在证明的局部展开循环假设定位成功。

## 4. 共享 bank 方差通过哪一个局部曲率起作用

本节固定数据并要求其精确 MLE 为内点：$q=q(t)\in(-R,R)$、$r(q)=t$。理想校准目标与精确似然在 q 的局部曲率为

$$
\gamma_q=J_t''(q)=b(q)r'(q)>0,\qquad
\mathcal I_q=\ell_t''(q)=\frac{r'(q)^2}{1-r(q)^2},
\qquad \gamma_q=\lambda(q)\mathcal I_q.
\tag{C.17}
$$

这里利用了驻点处 $r(q)-t=0$；不能把 $J_t''=br'$ 当成全域恒等式。

### 4.1 一个可证明的局部一阶展开

考虑一列可微扰动 $e_j$。假设 $J_t$ 在 q 邻域二阶连续可微，存在已定位的内点驻点 $\widehat\rho_j\to q$，满足 $(J_t+e_j)'(\widehat\rho_j)=0$，并有

$$
e_j'(q)=O_p(a_j),\qquad a_j\to0,
\qquad\sup_{I_0}|e_j''|=o_p(1).
\tag{C.18}
$$

那么

$$
\widehat\rho_j-q=-\frac{e_j'(q)}{\gamma_q}+o_p(a_j).
\tag{C.19}
$$

**证明。** 对驻点式在 q 展开，有 $0=e_j'(q)+\gamma_q+o_p(1) (historical reference not distributed)$。$J_t''$ 的连续性、已定位的一致收敛及 (C.18) 共同给括号中的余项；$\gamma_q>0$ 使该系数以高概率有统一正下界，从而先得差值 $O_p(a_j)$，再得 (C.19)。□

这是一条有明确局部条件的一阶展开，尚未证明当前有限 B 的输出满足这些条件。即使已有依概率展开，也不能未经一致可积性或尾部控制就对平方余项取期望，进而宣称实际风险的有限 B 速率或置信区间。

### 4.2 线性代理的二阶矩可以精确计算

对固定完整数据记真实逐 bank raw 路径导数为 $H_Z(q)$，$v_{\mathcal D}(q)=\operatorname{Var}_Z H_Z(q)$。所有观测共享 bank，故这里先对整批观测平均，不能把 $NB$ 项当作独立。校准池独立并固定后，令 $d_g(q)=\widehat g'(q)-g'(q)$，定义

$$
\xi_B(q)=B^{-1}\sum_bH_{Z_b}(q)-E_ZH_Z(q),\qquad
d_{lin}=-\frac{\xi_B(q)+d_g(q)}{\gamma_q}.
\tag{C.20}
$$

条件于数据和冻结校准，有精确的**线性代理**身份

$$
E[d_{lin}^2\mid\mathcal D,\mathcal C]
=\frac{d_g(q)^2+v_{\mathcal D}(q)/B}{\gamma_q^2},
\tag{C.21}
$$

以及

$$
E\!\left[\frac{\mathcal I_q}{2}d_{lin}^2\middle|\mathcal D,\mathcal C\right]
=\frac{d_g(q)^2+v_{\mathcal D}(q)/B}
{2b(q)^2[1-r(q)^2]}.
\tag{C.22}
$$

左侧只是将精确似然的局部二次形式应用于线性代理，**不等于实际输出的 KL**。该式解释为什么相同绝对梯度波动在较小 b 下可能对应更大的参数误差，也给出比较 $|d_g(q)|$ 与 $\sqrt{v_{\mathcal D}(q)/B}$ 的有量纲参照。确定加性校准改变均值、不改变这个固定点的训练 bank 方差；更换边际化评分则会改变方差。

重新平均独立校准池时，(C.21)—(C.22) 的分子应将 $d_g(q)^2$ 换成 $[E\widehat g'(q)-g'(q)]^2+\operatorname{Var}_{\mathcal C}(\widehat g'(q))$。只有一个共享校准池的训练区间不包含该方差。

对 clipped 边界 MLE，(C.17)—(C.22) 的内点驻点线性化不直接适用，应保留约束和非零的一阶项。可用第3节的界和边界标记，而不是将边界梯度均值全部算成校准偏差。本节没有估计新 iid 数据/共享 bank 的三个无条件 ANOVA 分量，也没有优化后风险的因果百分比。

若实验评价点只是参考表数值反演 $\widetilde q$，而非这里的精确 q，则平均理想校准梯度还含 $b(\widetilde q)[r(\widetilde q)-t]$。该项需要随参考误差保留，不能不经检查就归入校准曲线误差或宣称为零。

## 5. 与已证明的消退目标尺度衔接

这里只考虑既有渐近定理的适用序列：固定 a=1、R和相关块，$n\to\infty$，$2\le K_n\le Cn^p$，令

$$
w_n=\frac{J_{K_n}}{\sqrt{n\operatorname{Var}(\log\sigma(Z))}},\qquad A=1/4.
\tag{C.23}
$$

旧 有限校准桥接 (historical reference not distributed) F.16将一致梯度定理应用于两个合法固定教师，已得 $V'/w_n\to-A$ 一致。因此

$$
\sup_{\rho\in I}\left|\frac{b_n(\rho)}{w_n}-A\right|\to0,
\qquad b_{*,n}\asymp w_n,\qquad\lambda_{*,n}\asymp w_n.
\tag{C.24}
$$

这些比较常数依赖固定发射和参数区间，不依赖经验 t。于是第2节给出以下**充分条件**：

$$
\eta_n+\operatorname{osc}e_{MC,n}+\operatorname{osc}e_{g,n}=o_p(w_n)
\quad\Longrightarrow\quad
\ell_{t_n}(\widehat\rho_n)-\ell_{t_n}(q(t_n))\to_p0,
\quad |\widehat\rho_n-q(t_n)|\to_p0.
\tag{C.25}
$$

例如 $\eta_n=o_p(w_n)$、$\|e_{MC,n}\|_\infty=o_p(w_n)$、$\|\widehat g_n'-g_n'\|_\infty=o_p(w_n)$ 足够。这里校准误差比较的是理想**有限 n/K**校准；渐近校准若要使用该条件，应借助已经成立的一致 C¹ 余项，不能在当前 n=62或254直接假定相对误差很小。

对导数合同，若 $\|e_{MC,n}'\|_\infty+\|e_{g,n}'\|_\infty+\kappa_n=o_p(w_n)$，则 (C.14) 同样给出参数及精确经验超额收敛。对于只满足函数值近优而未认证 KKT 残差的输出，应使用 (C.25)，不能把两种求解合同混合。

若另有 iid 教师数据的 $N_n\to\infty$，则 $t_n\to_p r(\theta)$，(C.12) 进一步给完整观察 joint KL 趋零。此结论依赖上述误差事件确实成立，没有从单点 SE 自动推出统一MC误差、没有给未知结构样本复杂度，也没有证明现有有限 B 的风险按 $1/B$ 下降。

任意 $o(1)$ 近优误差、固定绝对校准导数误差或仅增加标签数，均不自动满足消退尺度要求；(C.25)是一条充分误差传递，不声称这些条件全为必要条件。当前四格有限实验仍需各自检查校准、评分和求解精度，不能按渐近关系直接合并结果。

## 6. 对新实验可直接使用的解释合同

- 用精确经验似然超额及到同数据 MLE 的距离分开描述计算误差；不要把到真教师的全部风险都称为校准误差。
- 使用统一绝对误差前，先说明各 n/K 的 b 或正斜率比是否可比较。没有严格 b 下界时，以上解析式只是有条件接口。
- 全域函数误差控制适用于近优输出；全域导数误差加约束残差给更紧界；局部导数误差则先需定位。这三种合同分别报告。
- 梯度方差除以局部曲率平方是线性代理诊断。只有另行验证定位、光滑扰动及余项控制后，才能把它提升为实际优化器的渐近结果；本轮不预设这一提升。
- 所有新结果应在独立数据和冻结规则下评价。旧小预算数据或教师风险不能回流选择本包校准规则、步长或统计门。
