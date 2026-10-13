# 独立教师的K2六阶展开：两个无关坐标可翻转总体偏好

本页是独立解析核算，基准 `04f6f0e09e3aff396d0b3651ae3d158fc4ed4662`。不修改旧包，不运行新Gaussian积分、随机样本或拟合。结论固定总维度D和候选相关区间，再令发射尺度a趋零；它不认证a=.05或a=1已经小于理论阈值，也不对增长D声称统一余项。

## 1. 正确指定模型与结论

固定整数 $D=n+2\ge2$、$0<R<1$。教师潜变量为 $N(0,I_D)$，给定潜变量后独立Bernoulli发射 $\sigma(aX_i)$。因此对每个a，观察教师 $P_a$ 都精确等于D位独立公平分布。候选均值为0、基础协方差为

\[
C_\rho=I_D+\rho(e_1e_2^\top+e_2e_1^\top),\qquad\rho\in[-R,R].
\tag{NP1}
\]

候选使用同一尺度a，基础谱位于 $[1-R,1+R]$；教师rho=0属于候选类。实际logit谱是a²乘该窗口，所以这里的弱发射极限不是固定正floor的高维结论。

取两个独立prior粒子，并对完整教师标签和粒子共同取期望：

\[
F_{D,a}(\rho)=E_{Y\sim P_a}E_{X_1,X_2\stackrel{iid}{\sim}N(0,C_\rho)}
\left[-\log\left\{\frac12\sum_{k=1}^2\prod_{i=1}^D\sigma(aY_iX_{ki})\right\}\right].
\tag{NP2}
\]

**定理。** 固定D与R，存在 $a_0,M>0$，使 $|a|\le a_0$ 时

\[
\boxed{F_{D,a}(\rho)-F_{D,a}(0)
=\frac{10-3D}{512}a^6\rho^2+\mathcal R_{D,a}(\rho),}
\tag{NP3}
\]

且

\[
\sup_{|\rho|\le R}\left(|\mathcal R_{D,a}(\rho)|
+|\partial_\rho\mathcal R_{D,a}(\rho)|
+|\partial_\rho^2\mathcal R_{D,a}(\rho)|\right)\le M|a|^8.
\tag{NP4}
\]

因此对充分小且非零a：

- D2或D3时，实际总体目标严格凸且偶对称，唯一全局极小点为rho=0，积分观察KL精确为0。
- D≥4时，实际总体目标严格凹且偶对称，恰有两个全局极小点rho=±R。每个极小点的完整观察KL为 $a^4R^2/32+O(a^6)$，对充分小正a严格为正。

这里“实际”指原NP2目标，不仅是六阶截断多项式。D2/3/4的六阶系数分别为 $1/128,1/512,-1/256$：添加第二个独立nuisance坐标使领先相关惩罚由正变负。该结论只涉及K2，不推出其它K的高阶规律或一般维度一致阈值。

## 2. 先证明可交换与一致余项

在rho区间的一个稍大紧邻域内，所有Gaussian协方差仍严格正定且具有统一上下谱界。对每个固定标签与粒子，令
$\ell_k(a)=\sum_i\log\sigma(aY_iX_{ki})$。bank损失为 $-\log(e^{\ell_1}+e^{\ell_2})+\log2$。

log-sigmoid的各正阶导数有界；log-sum-exp对其输入的各固定阶导数由softmax权重的有限多项式组成，因而有界。对a作r次微分（$1\le r\le8$），绝对值由固定常数乘 $(1+\|X_1\|+\|X_2\|)^r$ 控制。零阶损失也由一个线性Gaussian包络控制。

Gaussian密度对rho的一、二阶导数分别是密度乘以二阶、四阶多项式，系数受紧谱界控制。因此所有所需a八阶、rho两阶混合导数都由一个可积Gaussian多项式包络控制，导数与期望可以交换，且相应导数一致有界。教师分布在本页精确均匀，不随a变化。

Gaussian全局符号对称性使F及其rho导数均为a的偶函数。对F以及其rho一、二阶导数作七阶Taylor展开，余项为一致 $O(a^8)$。这不要求随机log-likelihood的局部幂级数在所有Gaussian实现上一致收敛；下文只是计算原目标在a=0的合法Taylor系数。

## 3. 将标签平均化为相关符号平均

对固定Y作变量变换 $Z_{ki}=Y_iX_{ki}$。候选基础协方差只改变第1、2位的非对角符号，成为 $C_{s\rho}$，其中 $s=Y_1Y_2\in\{-1,1\}$。独立公平教师使s的两种取值等权，其余标签符号不影响候选分布。

因此可以计算全正标签的bank损失，再对rho与−rho平均。以下期望带上横线表示这个符号平均；只保留相对于rho=0变化的项。

## 4. K2精确分解及六阶所需矩

对每个bank定义

\[
L_k=\sum_iX_{ki},\quad Q_k=\sum_iX_{ki}^2,\quad R_k=\sum_iX_{ki}^4,
\qquad X=L_1-L_2,\ Y=Q_1-Q_2,\ Z=R_1-R_2.
\tag{NP5}
\]

利用 $\log\sigma(x)=x/2-\log[2\cosh(x/2)]$，单bank有

\[
\ell_k=-D\log2+\frac a2L_k-\frac{a^2}{8}Q_k+\frac{a^4}{192}R_k+O(a^6).
\tag{NP6}
\]

K2损失精确分解为

\[
-\log\{(e^{\ell_1}+e^{\ell_2})/2\}
=-(\ell_1+\ell_2)/2-\log\cosh\{(\ell_1-\ell_2)/2\}.
\tag{NP7}
\]

第一项期望只取决于固定Gaussian单坐标边际，所以其全部rho相关系数为零。将NP6的差代入

\[
\log\cosh(t/2)=\frac{t^2}{8}-\frac{t^4}{192}+\frac{t^6}{2880}+O(t^8),
\]

Gaussian奇数总次数矩为零。原raw目标的a²、a⁴和a⁶系数中，所有可能随rho变化的部分分别为

\[
-\frac{E X^2}{32},\qquad
-\frac{E Y^2}{512}+\frac{E X^4}{3072},
\tag{NP8}
\]
\[
\frac{E(YZ)}{6144}+\frac{E(X^2Y^2)}{8192}-\frac{E X^6}{184320}.
\tag{NP9}
\]

注意NP9整体正负号来自NP7中负的logcosh；遗漏该负号会反转padding结论。

## 5. 独立Gaussian矩核算

对任一单位对角Gaussian协方差C，记
$S=\mathbf1^TC\mathbf1$、$T_2=\operatorname{tr}C^2$、$S_3=\mathbf1^TC^3\mathbf1$。两bank独立，直接Gaussian矩给

\[
EX^2=2S,\quad EY^2=4T_2,\quad EX^4=12S^2,\quad EX^6=120S^3.
\tag{NP10}
\]

又 $\operatorname{Cov}(X_i^2,X_j^4)=12C_{ij}^2$，故

\[
E(YZ)=2\operatorname{Cov}(Q_k,R_k)=24T_2.
\tag{NP11}
\]

为核混合项，把两bank堆叠为 $W\sim N(0,\Sigma)$，其中 $\Sigma=C\oplus C$，令 $v=(\mathbf1,-\mathbf1)$、$A=I_D\oplus(-I_D)$。此时 $X=v^TW$、$Y=W^TAW$、$\operatorname{tr}(A\Sigma)=0$。Gaussian矩配对或矩母函数微分给

\[
E[(v^TW)^2(W^TAW)^2]
=2(v^T\Sigma v)\operatorname{tr}(A\Sigma A\Sigma)
+8v^T\Sigma A\Sigma A\Sigma v,
\]

所以

\[
E(X^2Y^2)=8ST_2+16S_3.
\tag{NP12}
\]

对NP1，

\[
S=D+2\rho,\quad T_2=D+2\rho^2,\quad
S_3=D+6\rho+6\rho^2+2\rho^3.
\tag{NP13}
\]

对正负rho平均后，相对rho=0，a²所需矩的变化为0；a⁴两项分别给 $-8\rho^2/512$ 和 $48\rho^2/3072$，精确抵消。a⁶的三个矩变化为

\[
\Delta\overline{E(YZ)}=48\rho^2,\qquad
\Delta\overline{E(X^2Y^2)}=16(D+6)\rho^2,\qquad
\Delta\overline{EX^6}=1440D\rho^2.
\tag{NP14}
\]

代入NP9，得到

\[
\left(\frac1{128}+\frac{D+6}{512}-\frac D{128}\right)\rho^2
=\frac{10-3D}{512}\rho^2,
\]

结合第2节的一致余项，即证NP3–NP4。

## 6. 实际极小点与观察风险

令 $c_D=(10-3D)/512$。因为整数D≥2时 $c_D\ne0$，NP4给

\[
F''_{D,a}(\rho)=2c_Da^6+O(a^8)
\]

在整个相关区间上一致成立。故足够小非零a时，其符号等于 $c_D$。独立教师的正负相关平均使原目标精确偶对称：严格凸时唯一最小点0；严格凹时只能在两端取最小，且两端因偶对称具有相同值。这里没有用数值网格或有限差分推断全局性。

观察分布仅在active两位不同。候选自旋pair矩满足

\[
r_a(\rho)=E\tanh(aX_1/2)\tanh(aX_2/2)
=\frac{a^2\rho}{4}+O(a^4).
\]

对独立教师，同/异标签概率均为1/2；完整KL精确是

\[
\operatorname{kl}\left(\frac12\middle\Vert\frac{1+r_a(\rho)}2\right)
=-\frac12\log(1-r_a(\rho)^2)
=\frac{a^4\rho^2}{32}+O(a^6).
\tag{NP15}
\]

取rho=±R即得定理的风险后果。增加无关坐标没有增加此完整KL，因为其教师/候选因子完全相同；它也不保证平均全D条件查询风险或总count事件有同样大小的损失。

最后，raw优化的是随机K粒子预测器的期望log风险，而NP15评价积分后的观察分布。该区别贯穿本定理；“积分风险更高”不等于“raw在其自身随机预测合同上更差”。本页不提供任意有限a阈值、维度一致余项、有限数据学习率、K2修复算法或K4高阶结论。
