# 未截断膨胀下的 K 依赖收缩推论

本页是 WEAK_THEORY (historical reference not distributed) 的直接解析推论，0 新科学积分、随机样本或拟合。收缩只用 K 和所得候选矩阵，不输入教师参数；不将凸收缩重新命名为新通用算法。

## 条件和操作

沿用固定 d、固定 $K>2$、$\operatorname{diag}T=\mathbf1$ 的正确指定模型。候选集合 $\mathcal C$ 是 WEAK_THEORY W2 的谱紧凸类，或满足同一谱界且含 I 的非空凸闭子类。**额外要求未截断的膨胀矩阵**

$$
C_\gamma=I+\frac{K}{K-2}\operatorname{off}(T)
\quad\text{属于}\quad\mathcal C.
\tag{CR1}
$$

允许它在边界。W16 的投影定理在此给原始总体目标实际唯一极小点 $C_a=C_\gamma+O(a^2)$。令

$$
\alpha_K=\frac{K-2}{K}\in(0,1),\qquad
C_a^{\rm corr}=I+\alpha_K(C_a-I).
\tag{CR2}
$$

因为 $C_a^{corr}=(1-\alpha_K)I+\alpha_K C_a$，且 I 与 $C_a$ 都可行，凸性保证校正矩阵始终可行。没有在收缩后另作数据依赖 PSD 修正。由 CR1，

$$
C_a^{\rm corr}-T
=\alpha_K(C_a-C_\gamma)=O(a^2).
\tag{CR3}
$$

## 观察风险的八阶上界

**推论。** 在上述条件下，存在只依赖固定模型范围、d、K 的有限常数，使充分小 a 满足

$$
\boxed{D(P_a\Vert Q_{a,C_a^{\rm corr}})=O(a^8).}
\tag{CR4}
$$

**证明。** 对每个有限标签 $y\in\{-1,1\}^d$，Price 的对称非对角坐标导数给

$$
\partial_{C_{ij}}Q_{a,C}(y)
=a^2y_iy_jE\left[
\sigma'(ay_iX_i)\sigma'(ay_jX_j)
\prod_{l\ne i,j}\sigma(ay_lX_l)\right],\qquad
|\partial_{C_{ij}}Q_{a,C}(y)|\le a^2/16.
\tag{CR5}
$$

令 $p=d(d-1)/2$ 为非对角坐标数。沿连接 T 和任意 C 的可行线段积分，得到

$$
\max_y|Q_{a,T}(y)-Q_{a,C}(y)|
\le\frac{a^2\sqrt p}{16}\|c-t\|.
$$

所有候选固定 Gaussian 边际为标准正态，Jensen 和 $\log\cosh x\le x^2/2$ 给显式下界

$$
Q_{a,C}(y)\ge\exp\{E\log p_a(y\mid X)\}
=e^{-d\ell(a)}\ge2^{-d}e^{-da^2/8}.
$$

因此 KL 的 chi-square 上界给一个可明确核对的参数风险不等式：

$$
\boxed{
D(Q_{a,T}\Vert Q_{a,C})
\le\frac{4^d p}{256}e^{da^2/8}a^4\|c-t\|^2.
}
\tag{CR6}
$$

将 CR3 代入 CR6，并使用正确指定身份 $P_a=Q_{a,T}$，就得到 CR4。这里固定 d，常数含 $4^d$；没有对增长 d 声称同一常数。不能仅将 CR3 代入 W18 的四阶公式就宣称八阶，因为 W18 的 $O(a^6)$ 余项本身不够小；CR5–CR6 是必要的额外局部控制。□

## 学习解释和限制

若在固定正 a 和固定 K 下，理想经验全域近优点 $\widehat C_{N,B}$ 按 WEAK 定理3趋于 $C_a$，则同一确定性收缩应用于 $\widehat C_{N,B}$ 后趋于 $C_a^{corr}$；其风险由连续性趋于 CR4 的总体校正风险。该结论仍需要原全域优化与数值误差前提，不由固定多起点或 SGD 自动认证。固定 a 时 CR4 只是小量上界，不是精确零风险或精确无偏。

如果膨胀目标被约束投影截断，$C_a$ 的主极限是投影 $C_0$，CR2 一般不把 $C_0$ 送回 T，CR3–CR4 不能沿用。额外凸子类若不含 I，连收缩可行性都须另核。K2 的系数退化，本文不提供 K2 修复；a=1、高维旧族和原固定正 floor 结果也不在此弱发射推论内。

对于二维 $\theta_*=.15$、$\rho\in[-.75,.75]$ 的例子，K3/4/8 的膨胀目标均未被截断，校正操作分别将学得 rho 乘以 1/3、1/2、3/4。这只是已证明系数的可操作含义；截至本页，没有据新的数值结果筛选修正系数或宣称方法首创。

## 截断条件不可省略的解析边界例

仅作解析对照，仍取二维候选区间 $[-.75,.75]$，但令教师 $\theta_*=.75$ 位于上端点。这不是 NUMERIC 新增的科学教师。对任意固定 $K>2$，四阶主目标的导数为

$$
q'(\rho)=2A_K(\rho-\gamma_K\theta_*)
\le2A_K\theta_*(1-\gamma_K)<0.
$$

WEAK 的 $C^1$ 一致余项使充分小非零 a 下实际 raw 目标在整个区间严格递减。因此实际极小点恰为真参数 $\rho_a=\theta_*$，观察 KL 精确为零；膨胀目标在盒外，投影恰好送回真参数。

若此时仍应用 CR2，会得到 $\rho^{corr}=\alpha_K\theta_*\ne\theta_*$。由 WEAK W18，

$$
D(P_a\Vert Q_{a,\rho^{corr}})
=\frac{\theta_*^2}{8K^2}a^4+O(a^6)>0
$$

对充分小正 a 成立。故该收缩不是普遍改进；“未受投影截断”是实质条件，不是技术性装饰。该例没有引入新的实验、定位程序或数值参数搜索。
