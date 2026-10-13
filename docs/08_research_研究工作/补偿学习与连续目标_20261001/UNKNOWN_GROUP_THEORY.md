# 未知两组的正公共因子补偿学习：split ROW 与矩估计

这是独立于本轮已冻结 RUN 与 T 的新增解析桥接，不修改它们，也不运行新积分、随机样本或拟合。正族 ROW 的符号和方差常数在本文重新证明，不直接继承旧负族结论。所有风险都是完整 d 位向量的总 forward KL，未除以 d。

## 1. 参数类、已知信息与算法

固定已知范围

$$
0<a<b<\infty,\qquad 0<c_-\le c_+<\infty,
\qquad 0<\delta\le b-a.
$$

偶数 $d=2m$，高尺度坐标集合 $H\subset\{1,\ldots,d\}$ 满足 $|H|=m$。未知参数是 H、$c\in[c_-,c_+]$ 与
$a\le s_L<s_H\le b$，其中 $s_H-s_L\ge\delta$。教师是零均值正公共因子模型：

$$
\rho=\frac{c}{d-1},\quad
\eta_i^\star=s_i^\star\{\sqrt\rho Z+\sqrt{1-\rho}E_i\},\quad
s_i^\star=s_H\mathbf1_{i\in H}+s_L\mathbf1_{i\notin H}.
\tag{UG1}
$$

所有 Z、E_i 独立标准 Gaussian，条件二元发射独立且成功概率为 $\sigma(\eta_i^\star)$。记 $Y_i=2C_i-1$，观察律为 P。各自旋边际为零，但同一向量内的坐标及 pair 不独立。

置

$$
\alpha(s)=E[Z\tanh(sZ/2)],\quad A_0=\alpha(a),\quad
R=\frac{\alpha(b)}{A_0},\qquad
0<L\le\sqrt{c_-},\quad U\ge\sqrt{c_+}R.
\tag{UG2}
$$

算法只用预定盒 [L,U]、已知 floor a 的常数 A_0、组大小 m 和训练观察；未知 c、教师尺度、教师 PMF 与真实 H 都不输入算法。定义固定正数

$$
h_0=\alpha(a)E[Z^2\operatorname{sech}^2(bZ/2)],\qquad
\Delta_0=h_0\delta,\qquad \kappa=\frac{c_-\Delta_0}{6}.
\tag{UG3}
$$

以下取足够大的偶数 d，使

$$
d\ge4,\qquad d-1\ge2U^2,\qquad
\left(\frac{c_+}{d-1}\right)^2\le\frac{\Delta_0}{2}.
\tag{UG4}
$$

这些条件只涉及已知范围，保证所有相关和载荷合法、$\rho\le1/2$ 及候选载荷平方至多 1/2。不未经核对将其自动套入某个有限实验维数。

取互相独立的两批 iid 完整标记向量，大小为 $n_1,n_2\ge1$。

第一批累积**正号** ROW 统计

$$
\widehat b_i=\frac1{n_1}\sum_{n=1}^{n_1}
Y_i^{(n)}\sum_{j\ne i}Y_j^{(n)},
\tag{UG5}
$$

取 m 个最大值作为 $\widehat H$，平局用事前固定的坐标顺序。旧负族使用负号，不能照搬到此处。

第二批对 $G\in\{\widehat H,\widehat H^c\}$ 计算

$$
T_G^{(n)}=m^{-1/2}\sum_{i\in G}Y_i^{(n)},\qquad
\widehat z_G=\frac{d-1}{(m-1)A_0^2}
\left\{\frac1{n_2}\sum_n(T_G^{(n)})^2-1\right\},
\qquad
\widehat w_G=\sqrt{\Pi_{[L^2,U^2]}(\widehat z_G)}.
\tag{UG6}
$$

把 $\widehat w_G$ 放回相应坐标，记为 $\widehat u_i$。不按第二批估计载荷的大小再重新标记坐标。输出是精确积分观察模型

$$
\widehat\eta_i=a\left\{
\frac{\widehat u_i}{\sqrt{d-1}}Z+
\sqrt{1-\frac{\widehat u_i^2}{d-1}}\,\widetilde E_i
\right\},\qquad Q_{\widehat u}=\operatorname{Law}(C).
\tag{UG7}
$$

即使分组或矩估计错误，输出也在固定盒内且有统一严格正谱界。目标是学习有效响应载荷，不是分别识别 c 和两个原尺度。

## 2. 正族 pair 界与 ROW 恢复

对正相关为 rho 的两个标准 Gaussian，令
$r_{st}=E[\tanh(sX/2)\tanh(tX'/2)]$。在正交归一 Hermite 基下，奇函数只有奇阶系数 $a_j(s)$，且 $a_1(s)=\alpha(s)$、$\sum_j a_j(s)^2\le1$。所以

$$
|r_{st}|=\left|\sum_{j\ge1,\,j\text{ odd}}a_j(s)a_j(t)\rho^j\right|
\le\rho\left(\sum_j a_j(s)^2\right)^{1/2}
\left(\sum_j a_j(t)^2\right)^{1/2}\le\rho.
\tag{UG8}
$$

正性也可直接核对：条件于正公共 Gaussian Z 后，两个条件自旋均值都是关于 Z 的奇、严格递增函数，故乘积非负。因此本族所有不同坐标有 $0\le E(Y_iY_j)\le\rho$。这里只使用 Hermite Cauchy 界，没有把一般 pair 或坐标当成独立。

高低组的总体 ROW 均值为

$$
b_H=(m-1)r_{HH}+mr_{HL},\qquad
b_L=(m-1)r_{LL}+mr_{HL},\qquad
\mu=b_H-b_L=(m-1)(r_{HH}-r_{LL}).
\tag{UG9}
$$

有 $\alpha'(s)=\frac12E[Z^2\operatorname{sech}^2(sZ/2)]>0$，且在 [a,b] 上 $2\alpha(s)\alpha'(s)\ge h_0$。故
$\alpha(s_H)^2-\alpha(s_L)^2\ge\Delta_0$。各尺度 Hermite 的三阶以上平方系数余项都在 $[0,\rho^3]$，由此

$$
r_{HH}-r_{LL}\ge\rho\Delta_0-\rho^3
\ge\frac{\rho\Delta_0}{2},\qquad
\mu\ge\frac{c\Delta_0(d-2)}{4(d-1)}\ge\kappa.
\tag{UG10}
$$

最后两步分别使用 UG4 和 $d\ge4$。这给整个未知参数类的统一正间隔。

固定真高低坐标对 i、j，单观察的 ROW 差为

$$
W=(Y_i-Y_j)\sum_{k\ne i,j}Y_k,
\qquad EW=\mu.
$$

交叉项 $Y_iY_j$ 在两行相减时精确抵消。由 UG8，

$$
\operatorname{Var}W\le EW^2
\le4E\left(\sum_{k\ne i,j}Y_k\right)^2
\le4\{(d-2)+(d-2)(d-3)\rho\}
\le4d(1+c_+).
\tag{UG11}
$$

正 pair 不能像负族那样删除；这是必须重证的方差步骤。同时 $|W|\le2(d-2)$，从而 $|W-EW|\le4d$。对 iid 的 n1 个完整观察用中心化 Bernstein，并把事件 $\overline W\le0$ 包含于 $\overline W-EW\le-\kappa$，得到

$$
P(\widehat b_i\le\widehat b_j)
\le\exp\left[-\frac{n_1\kappa^2}
{8d(1+c_++\kappa/3)}\right].
$$

对 $m^2$ 个跨组比较作 union bound；不需要这些比较互相独立。于是

$$
\boxed{
P(\widehat H\ne H)\le
\min\left\{1,\frac{d^2}{4}
\exp\left[-\frac{n_1\kappa^2}{8d(1+c_++\kappa/3)}\right]\right\}.
}
\tag{UG12}
$$

这是标准 Bernstein 加 ROW 的特化，不申报新的通用聚类算法或最优恢复率。

## 3. 任意盒内坐标载荷的完整观察风险

为处理错误分组，不能只对真组共享参数引用风险式。以下允许任意 $u=(u_1,\ldots,u_d)\in[L,U]^d$，并用 UG7 定义 $Q_u$。记
$w_i^\star=\sqrt c\,\alpha(s_i^\star)/A_0\in[L,U]$。定义固定常数

$$
m_1=\sqrt{2/\pi},\quad E_a=e^{2am_1+2a^2},\quad
L_1=\frac{m_1}{2}(b^2+R^3a^2),\quad
L_3=\frac{b^3+R^3a^3}{24},
$$
$$
C_0=\sqrt2E_a(2L_1^2+24L_1L_3+120L_3^2),\qquad
C_1=a^2\sqrt2E_a(2+m_1^2).
\tag{UG13}
$$

**引理。** 对每个盒内 u，不论它是否按真组排列，

$$
\boxed{
D(P\Vert Q_u)\le2C_0d\rho^3+
\frac{C_1}{d-1}\sum_{i=1}^d(u_i-w_i^\star)^2.
}
\tag{UG14}
$$

证明逐坐标进行。条件于同一个公共值 z，教师成功概率记为 $p_i(z)$，floor 载荷 u 的概率记为 $q_{u_i}(z)$。对 $f_s(x)=\tanh(sx/2)$ 在私噪声处二阶 Taylor 展开，利用奇性、$\|f_s''\|_\infty\le s^2/2$、$\|f_s'''\|_\infty\le s^3/4$，得到全实轴余项

$$
|E f_s(\sqrt t z+\sqrt{1-t}E)-\sqrt t\,\alpha(s)z|
\le t^{3/2}\{s^2m_1|z|/2+s^3|z|^3/24\},\quad0\le t\le1/2.
$$

比较载荷 $w_i^\star$ 使一阶项精确消去，所以

$$
|p_i(z)-q_{w_i^\star}(z)|
\le\tfrac12\rho^{3/2}(L_1|z|+L_3|z|^3).
\tag{UG15}
$$

对 $\lambda=u_i/\sqrt{d-1}$ 的路径微分及 $\lambda^2\le1/2$ 给

$$
|\partial_\lambda q_\lambda(z)|\le\frac a4(|z|+m_1),\qquad
|q_{u_i}(z)-q_{w_i^\star}(z)|
\le\frac{a|u_i-w_i^\star|}{4\sqrt{d-1}}(|z|+m_1).
\tag{UG16}
$$

整个连接线段都在合法盒内。由 $\sigma(x)\ge e^{-|x|}/2$ 和 Jensen，$q_{u_i}$ 及其补概率均至少为 $\frac12e^{-a|\lambda z|-am_1}$，从而

$$
\frac1{q_{u_i}(1-q_{u_i})}\le4E_a e^{z^2/4}.
$$

将 UG15–UG16 的误差平方相加上界（系数各为 2），代入 Bernoulli KL 的 chi-square 上界；再用
$E[Z^{2j}e^{Z^2/4}]=\sqrt2\,2^j(2j-1)!!$，得到单坐标期望界
$2C_0\rho^3+C_1(u_i-w_i^\star)^2/(d-1)$。条件于 Z 的坐标独立使联合 $(Z,C)$ KL 相加，随后边际化 Z 的 DPI 给 UG14。没有使用 KL 三角不等式，也没有假定错误分组下的坐标共享同一个真尺度。

因为每个 u_i 与 $w_i^\star$ 都在 [L,U]，且 $d\ge4$，可取与维数和未知参数无关的

$$
\boxed{
M=\frac8{27}C_0c_+^3+\frac43C_1(U-L)^2
\quad\text{使}\quad D(P\Vert Q_u)\le M.
}
\tag{UG17}
$$

这里用了 $d/(d-1)^3\le4/27$ 和 $d/(d-1)\le4/3$。因此分错组的损失无需以 O(d) 粗界控制，但这个更强界依赖当前弱正公共因子和盒内载荷结构。

## 4. 独立第二批矩估计与风险分账

令 $\mathcal G=\{\widehat H=H\}$。它只依赖第一批。条件于第一批及 $\mathcal G$，第二批仍是原模型的 iid 观察，UG6 使用的两个组正是真组。

为自足核对第二批速率，令 $A_T=1+b^2c_+/6$。给定公共 Z，真组标准化和 $T_g$ 分成条件中心 Bernoulli 和与 $\sqrt m\,E(Y_i\mid Z)$。前者的条件 MGF 至多 $e^{t^2/2}$；后者是零均值 Gaussian Lipschitz 函数，Lipschitz 平方至多 $mb^2\rho/4\le b^2c_+/6$。依次取条件期望，得到 $Ee^{tT_g}\le e^{A_Tt^2/2}$，以及 $ET_g^4\le16A_T^2$，不要求两部分独立。

同组精确 pair 矩的 Hermite 展开给
$0\le r_g-\rho\alpha(s_g)^2\le\rho^3$。由 $(d-1)/(m-1)\le3$，UG6 的原始平方载荷偏差至多 $c_+^3/[A_0^2(d-1)^2]$，方差至多 $144A_T^2/(A_0^4n_2)$。投影不增加到盒内真值的距离，平方根在 $[L^2,U^2]$ 上为 $1/(2L)$-Lipschitz，故

$$
E(\widehat w_g-w_g^\star)^2\le
e_{n_2,d}:=\frac{144A_T^2/n_2+c_+^6/(d-1)^4}{4L^2A_0^4}.
\tag{UG18}
$$

该陈述在真组事件上使用独立第二批；不将经同批选组后的矩当作无选择偏差的样本矩。

对好事件使用 UG14、UG18；对坏事件逐数据实现使用 UG17。得到完整的统一风险界：

$$
\boxed{
\begin{aligned}
\sup_{H,c,s_H,s_L}E D(P\Vert Q_{\widehat u})
&\le 2C_0c_+^3\frac d{(d-1)^3}
+\frac{2C_1m}{d-1}e_{n_2,d}\\
&\quad+M\min\left\{1,\frac{d^2}{4}
\exp\left[-\frac{n_1\kappa^2}{8d(1+c_++\kappa/3)}\right]\right\}.
\end{aligned}
}
\tag{UG19}
$$

期望覆盖两批完整向量，sup 覆盖第1节全部未知参数。这是当前估计器的风险定理，没有要求它最小化某个 likelihood。

特别地，可令

$$
C_a=16C_0c_+^3+\frac{C_1c_+^6}{3L^2A_0^4},\qquad
C_b=\frac{48C_1A_T^2}{L^2A_0^4},
$$

把 UG19 的前两项上界为 $C_ad^{-2}+C_bn_2^{-1}$。这里使用 $d/(d-1)^3\le8/d^2$、$d/(d-1)\le4/3$ 和 $(d-1)^4\ge d^2$。因而确有

$$
E D\le C(d^{-2}+n_2^{-1})+
M\frac{d^2}{4}e^{-n_1\kappa^2/[8d(1+c_++\kappa/3)]}.
\tag{UG20}
$$

## 5. 一条充分样本路径与边界

取 $n_2=d^2$，并令

$$
n_1\ge\left\lceil
\frac{8d(1+c_++\kappa/3)}{\kappa^2}
\{4\log d+\log(1+M)\}\right\rceil.
\tag{UG21}
$$

则坏事件项至多 $1/(4d^2)$，UG19 给 $ED\le(C_a+C_b+1/4)d^{-2}$。因此 $n_1=O(d\log d)$、$n_2=d^2$ 是期望总 KL 达到 $O(d^{-2})$ 的充分配置；不是必要样本下界或最优率。第一批可 O(n1 d) 累加行统计，排序 O(d log d)；第二批 O(n2 d) 算矩。样本计完整向量，bit 成本是 $(n_1+n_2)d$，没有把 pair 数当成独立样本数。

两个重要边界是：固定正 c 下界和尺度分离维持 ROW 间隔；已知有限上界维持统一方差与风险常数。若 c 趋零、尺度差趋零、符号未知、均值非零、组大小不等、缺失坐标或任意相关结构，本文不提供相同结论。非零边际尤其会改变未经中心化的 ROW 排序方向及量级。

矩算法学习有效响应载荷，不分别恢复 c 与两个教师尺度；不据此声称其它程序不能识别这些参数。教师在有限 d 的观察可表示性无需先决定，风险证明保留近似表示误差。数学中的 A0 为精确已知常数，实际数值 A0 及近似部署若有误差须另传播，旧数值资格不自动是严格证书。

截至本文尚无本阶段未知组新实验。已冻结 RUN 的 raw/col/准MLE 都在已知组合同下，不能因本桥接而改称已有未知组训练结果。也未证明 raw/col 在未知组类中达到全局最优，未证明同批 ROW+MOM、端到端 SGD 或一般模型保证。ROW、样本分割、Hermite 矩与投影都是经典工具，本文不主张新的通用算法首创。

来源：本轮 LEARNING_THEORY (historical reference not distributed) 的已知组矩与逐坐标风险步骤；旧 负族 ROW (historical reference not distributed) 只提供比较框架，正族符号与方差已在此重证；旧 [未知尺度 split 证明](../v2收束与v3方向_20260930/THEORY_UNKNOWN_SCALES.md) 只提供好坏事件分账思路，其已知 c 的尺度反演和 Gaussian KL 转换未被直接移植。本文独立新增，不回写任何旧冻结。
