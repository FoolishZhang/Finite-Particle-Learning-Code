# 有限标签、有限粒子集到总体风险的桥接

2026-10-01 解析稿，供独立数学审查。本文件只处理本包零均值正族的固定有限候选网格；不扩展到均值自由模型，不保证SGD。未运行新随机实验、拟合或积分。第8节仅计算显式界的数值，不是科学样本或目标积分。

## 1 对象和随机合同

固定偶数 d≥4、0<a<b、0<c<1、ρ=c/(d−1)，已知等大组 H。有限网格 G⊂[a,b]² 在标签和粒子之前固定，大小 M，包含真尺度 θ_star 及 (a,a)。所有候选均值0、相关矩阵 R⁺=(1−ρ)I+ρ11ᵀ。教师 P_star=Q_theta_star 正确指定，真尺度固定且均严格大于 a。候选实际部署分布 Q_theta 指精确积分模型。

Y_1,…,Y_N 独立同分布于 P_star。K 是每个粒子集的粒子数；B 是在 log 外平均的独立粒子集数。候选之间可以完全共用标准Gaussian随机数。

- **逐标签独立：**每个 (n,b) 使用独立的 K 粒子集 U_nb；共有 NB 个独立粒子集。
- **共享粒子集：**仅生成 B 个粒子集 U_b，全部 N 个标签与全部候选共用它们；粒子集仍独立于标签。

本轮主实验采用第二种合同。raw与col可共享公共因子，二者无需独立；联合概率界通过union bound处理。数据选择的候选网格不能直接用本定理中的实际 M，除非改用事前固定的全部可能候选集合。

单粒子损失记为 H_theta^x(y,U)，x∈{raw,col}。raw使用

$$
\eta_i=s_i\{\sqrt\rho Z+\sqrt{1-\rho}E_i\},\qquad
H_\theta^{\rm raw}(y,U)=-\sum_i\log\sigma(y_i\eta_i).
$$

col精确积分 E_i，置 p_s(z)=E_E σ(s[√ρz+√(1−ρ)E])，正式定义为

$$
H_\theta^{\rm col}(y,z)=
-\sum_i\left[\frac{1+y_i}{2}\log p_{s_i}(z)
+\frac{1-y_i}{2}\log(1-p_{s_i}(z))\right].
\tag{F1}
$$

每个粒子集的评分与总体目标为

$$
L_\theta^x(y,U_{1:K})=-\log\left[K^{-1}\sum_{k=1}^K
e^{-H_\theta^x(y,U_k)}\right],\quad
m_\theta^x(y)=E_U L_\theta^x(y,U_{1:K}),\quad
F_\theta^x=E_Ym_\theta^x(Y).
\tag{F2}
$$

经验目标分别是 (NB)^−1 Σ_nb L(Y_n,U_nb) 或 (NB)^−1 Σ_nb L(Y_n,U_b)。二者均无偏估计同一个 F_theta^x；不把 BK 个粒子合入同一个 log。

## 2 两评分共同的关键结构

可选取下列全网格Gaussian Lipschitz平方常数：

$$
A_{\rm raw}=b^2(1+c)d,\qquad
A_{\rm col}=b^2d^2\rho=\frac{b^2cd^2}{d-1}.
\tag{F3}
$$

raw常数来自原THEORY中的 h_y 梯度界与最大协方差特征值。col的常数不需要概率clip：

$$
p_s'(z)=s\sqrt\rho E\sigma'(s[\sqrt\rho z+\sqrt{1-\rho}E]),
\quad
\sigma'(t)\le\min\{\sigma(t),1-\sigma(t)\},
$$

故两种标签下的绝对 log-probability 导数均≤s√ρ。单核求和后的导数至多√ρ Σ_i s_i≤db√ρ。更细常数 A_col,theta=ρ(Σ_i s_i)² 亦可用，但下文用事前共同上界。

若单核对标准Gaussian输入为 √A-Lipschitz，粒子集损失对全部 K 个输入的梯度块是 π_k∇H_k，其中 π_k≥0、Σπ_k=1。因此梯度平方和≤AΣπ_k²≤A。**不能由 K 个独立粒子自动把该常数除以 K。**

两种单核的期望还都与标签取值无关，记为 Ψ_theta^x。raw为 Σ_i ell(s_i)，ell(s)=E log[2cosh(sZ/2)]。col因为 p_s(−z)=1−p_s(z)，有 E log p_s(Z)=E log(1−p_s(Z))，同样不依赖 y。这只用零均值对称性，不要求标签彼此独立或其边际恰为1/2。

由Gaussian集中与期望最大值界，对每个固定 y，

$$
\Psi_\theta^x-t_x\le m_\theta^x(y)\le\Psi_\theta^x,
\qquad t_x=\sqrt{2A_x\log K}.
\tag{F4}
$$

证明与原THEORY式(6)相同：逐粒子集评分位于单核损失最小值与平均值之间。K=1时 m_theta^x(y)=Ψ_theta^x，t_x=0。col的 Ψ 不是raw的 Ψ；不能用(F4)把col也直接套进raw尺度定位。

## 3 N/K/B明确分解的统一浓度

**定理F1。** 对任一评分 x，令 0<δ<1。以至少1−δ的概率，整个固定 M 点网格满足

$$
\max_{\theta\in G}|\widehat F_\theta^x-F_\theta^x|
\le e_x^{\rm ind}:=
\sqrt{\frac{2A_x}{N}\left[\frac1B+\frac{\log K}{2}\right]
\log\frac{2M}{\delta}}
\tag{F5}
$$

用于逐标签独立粒子集；共享粒子集则为

$$
e_x^{\rm shared}:=
\sqrt{2A_x\left[\frac1B+\frac{\log K}{2N}\right]
\log\frac{2M}{\delta}}.
\tag{F6}
$$

**证明。** 给定全部标签，经验损失对标准Gaussian粒子输入的Lipschitz平方分别≤A_x/(NB)与A_x/B。共享合同中，对每个粒子集先平均 N 个标签只能保证Lipschitz常数不增加；没有独立 N 次粒子随机性可供除以N。

两种合同的条件期望均为 N^−1 Σ_n m_theta^x(Y_n)。根据(F4)，单个 m_theta^x(Y) 落入长度 t_x 的固定区间，故条件均值的中心化MGF≤exp(λ²t_x²/(8N))。为自足起见，此Hoeffding引理可由指数倾斜下的方差≤t_x²/4、对 log-MGF 二次积分得到。

先条件Gaussian集中，再对标签积分，得到经验目标中心化MGF的方差代理

$$
v_x^{\rm ind}=\frac{A_x}{NB}+\frac{t_x^2}{4N},\qquad
v_x^{\rm shared}=\frac{A_x}{B}+\frac{t_x^2}{4N}.
$$

这里粒子残差与标签条件均值不需要独立。Chernoff界及 M 个候选的双侧union bound给(F5)–(F6)。候选间共享粒子不会改变union bound。□

若同时陈述raw和col，用各自δ/2，即把 log(2M/δ)换为 log(4M/δ)。提高 N 不能消除共享粒子集的1/B项；提高B不改变K。固定B、维度增长时，此界不提供一致性。

## 4 经验选点和数值误差

设实际计算目标 tilde F 满足整个网格的确定误差界

$$
\max_{\theta\in G}|\widetilde F_\theta-\widehat F_\theta|\le\eta,
\quad
\widetilde F_{\widehat\theta}\le\min_{\theta\in G}\widetilde F_\theta
+\varepsilon_{\rm alg},\qquad
\tau:=\varepsilon_{\rm alg}+2\eta.
\tag{F7}
$$

则精确经验目标的次优性≤τ；在F1事件上，

$$
F_{\widehat\theta}^x\le\min_{\theta\in G}F_\theta^x+2e_x+\tau.
\tag{F8}
$$

若η只在概率至少1−δ_num的事件上有效，最终失败概率另加δ_num。求积收敛检查本身不自动给(F7)的严格η。概率clip会改变评分；只有在另行控制其相对理想评分误差时才可代入。这里的近最优仅对固定有限网格，不是连续总体全局最优或SGD轨迹保证。

## 5 col从有限网格ERM到精确部署风险

**定理F2。** 在上述正确指定且 θ_star∈G 的合同下，经验col选点满足

$$
D(P_\star\Vert Q_{\widehat\theta}^{+})
\le C_c\omega_K+2e_{\rm col}+\varepsilon_{\rm alg}+2\eta,
\qquad C_c=\frac c{1-c},\quad K\ge2,
\tag{F9}
$$

概率至少1−δ（数值误差事件另记）。omega_K使用原THEORY式(28)的显式充分界：

$$
\omega_K=\frac{12\log K}{\sqrt K}
+6\sqrt{\frac2\pi}\frac{t_K+t_K^{-1}}K,
\qquad t_K=\sqrt{2\log K}.
$$

**证明。** Jensen给精确部署风险 R(P_star;θ)≤F_col(P_star;θ)。不必与连续最小值比较，直接用经验选点与网格中的真参数，得到

$$
R(P_\star;\widehat\theta)-H(P_\star)
\le F_{\widehat\theta}^{\rm col}-H(P_\star)
\le F_{\theta_\star}^{\rm col}-H(P_\star)+2e_{\rm col}+\tau.
$$

真参数处的偏差上界为 C_c omega_K，得到(F9)。没有假设错误候选的Jensen偏差满足同样上界，也没有声称参数恢复。□

如果网格不含真参数，右侧必须保留 min_{θ∈G}F_col(P_star;θ)−H(P_star)，不能仅以最小精确KL替换它。当前49格包含(6,2)，无需这个额外近似问题。任意事件的log-loss超额≤(F9)右端，scalar Brier超额≤其一半；这针对选出参数的精确 Q 部署，不包含另一个有限粒子预测器的推断误差。

## 6 raw的更直接有限数据尺度定位

raw有比(F8)更直接的条件标签结论。固定任意标签集，只要求它独立于粒子，记条件均值 m_{D,theta}=N^−1Σ_n m_theta(Y_n)。令

$$
r_{\rm raw}=
\sqrt{\frac{2A_{\rm raw}}{B_{\rm eff}}
\log\frac{2M}{\delta}},\qquad
B_{\rm eff}=NB\ \text{或}\ B,
$$

分别对应逐标签独立和共享合同。条件Gaussian集中使全网格经验目标与其条件均值之差≤r_raw，概率至少1−δ。此处无需假设标签为iid。

**定理F3。** 网格含(a,a)，实际程序满足(F7)时，在这个条件事件上

$$
(\widehat u-a)+(\widehat v-a)
\le\frac{2}{d\ell'(a)}
\left[t_{\rm raw}+2r_{\rm raw}+\varepsilon_{\rm alg}+2\eta\right].
\tag{F10}
$$

证明链为 Ψ_hat≤m_{D,hat}+t_raw≤hat F_hat+r_raw+t_raw≤hat F_floor+τ+r_raw+t_raw≤Ψ_floor+τ+2r_raw+t_raw，再用ell的凸性。它不需要先证明经验选点接近连续总体最优，也不需要增加N来控制标签抽样误差。共享粒子下，N增大不能替代B；但足够大的B已可使这个结论条件于任何固定标签成立。

## 7 一条可执行的多项式有限网格路径

取当前固定7点坐标轴及49格，包含floor(1,1)、truth(6,2)，其余尺度与1有固定正间距。沿偶数d→∞采用同一正族、同一标签、共享的公共因子，令

$$
K_d=d,\quad N_d=B_d=d^2,\quad\delta_d=d^{-2},\quad
\varepsilon_{{\rm alg},d}\to0,\quad\eta_d\to0.
\tag{F11}
$$

这里δ_d/2分别分配给定理F3的raw条件bank事件和定理F1的col全目标事件；分离只需这两个事件，未同时把F1的raw全目标事件算入该总δ_d。由于A_raw,A_col=O(d)，

$$
e_{\rm col}^{\rm shared}=O(\log d/\sqrt d),\qquad
r_{\rm raw}=O(\sqrt{\log d/d}),\qquad
t_{\rm raw}=O(\sqrt{d\log d}).
$$

(F9)使col的精确部署总KL趋零；(F10)使raw的尺度偏离和趋零。因为网格固定且floor隔离，充分大d时，raw在该高概率事件上实际选到floor。原THEORY的事件CLT因此给非消失Brier/log-loss后果。这是**有限标签、有限共享粒子集的网格程序分离**，不依赖连续全局最优或SGD假设。

还可得到期望col总KL趋零：任意候选精确部署有

$$
0\le D(P_\star\Vert Q_\theta)\le R(P_\star;\theta)
\le\Psi_\theta^{\rm raw}\le d\ell(b).
$$

故坏事件贡献≤δ_d d ell(b)→0；若数值误差事件另有失败概率，需其也为o(1/d)。raw的事件Brier有界于1，因而其期望亦趋于floor的正事件损失常数。所有结论以(F7)成立为前提。计算量只是多项式充分路径，未主张最优样本率、最优粒子率或相同运行成本。

## 8 本轮有限预算下，此界有多空

以下只将显式公式代入 M=49、a=1、b=8、c=.9、K=d；选择单个训练程序中两评分联合δ=.05，因而使用 log(4M/.05)。这不是整个实验所有格/流的同时家族。暂取ε_alg=η=0；结果只显示证明给出的上界，不是实测误差。

|d|N|B|e_raw|e_col|C_c omega_K|col总KL上界 C_c omega_K+2e_col|
|---:|---:|---:|---:|---:|---:|---:|
|16|256|16|46.760490|33.238228|82.344650|148.821106|
|16|256|128|20.636588|14.668872|82.344650|111.682393|
|16|4096|16|44.978840|31.971799|82.344650|146.288247|
|16|4096|128|16.199427|11.514855|82.344650|105.374359|
|64|256|16|95.366981|66.154928|58.319937|190.629793|
|64|256|128|45.300802|31.424621|58.319937|121.169180|
|64|4096|16|90.078728|62.486530|58.319937|183.292998|
|64|4096|128|32.733449|22.706800|58.319937|103.733537|

这些界不能认证本轮小的实测优化误差或事件收益；对应的Brier上界甚至超过其平凡最大值1。有限实验必须独立报告结果、数值资格和不确定性，不能借(F11)追认当前预算。N从256增到4096仅消减标签项；共享bank误差和保守的truth-bias界在这里仍占主导。

本轮col-only K=4d辅助格可代入一般公式，但不是与raw K=d同粒子预算的对照。数值MLE参考也有自己的求积误差，未由本证明自动认证。
